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
    import wheel as wh
    return ONE, wh

# Paths
DATA_ROOT = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
OUTPUT_DIR = "/space/scratch/SWR_final_pipeline/validation_data_figure9"
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
    return parser.parse_args()

def get_speed_data_abi(session_obj, dataset_type):
    """Get speed data for ABI datasets"""
    if dataset_type == "abi_visbehave":
        wheel_velocity = session_obj.running_speed['speed'].values
        wheel_time = session_obj.running_speed['timestamps'].values
    else:  # abi_viscoding
        wheel_velocity = session_obj.running_speed['speed'].values
        wheel_time = session_obj.running_speed['timestamps'].values
    
    # Interpolate to 1500 Hz
    interp_func = interp1d(wheel_time, wheel_velocity)
    wheel_time = np.linspace(wheel_time[0], wheel_time[-1], int(len(wheel_time) * 1500 / len(wheel_time)))
    wheel_velocity = interp_func(wheel_time)
    return wheel_velocity, wheel_time

def get_speed_data_ibl(one, session_id):
    """Get speed data for IBL dataset"""
    wheel = one.load_object(session_id, 'wheel', collection='alf')
    pos, t = wh.interpolate_position(wheel.timestamps, wheel.position)
    delta_t = 1/np.array([t[i]-t[i-1] for i in range(1, len(t))])
    wh_vel, wh_accel = wh.velocity_filtered(pos, delta_t)
    
    # Interpolate to 1500 Hz
    interp_func = interp1d(t[1:], wh_vel)
    wheel_time = np.linspace(t[1], t[-1], int(len(t) * 1500 / len(t)))
    wheel_velocity = interp_func(wheel_time)
    return wheel_velocity, wheel_time

def main():
    args = parse_args()
    dataset_type = args.dataset
    config = DATASET_CONFIGS[dataset_type]
    
    print(f"Processing dataset: {dataset_type}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    theta_filter = np.load(THETA_FILTER_PATH)
    theta_filter = theta_filter[list(theta_filter.keys())[0]]
    
    # Lazy load appropriate API
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
    
    # Data folders
    swr_dir = os.path.join(DATA_ROOT, config["swr_dir"])
    lfp_dir = os.path.join(DATA_ROOT, config["lfp_dir"])
    session_dirs = sorted(os.listdir(lfp_dir))
    if args.max_sessions:
        session_dirs = session_dirs[:args.max_sessions]
    
    results = []
    for session in session_dirs:
        print(f"  Session: {session}")
        session_id = session.split('_')[-1]
        lfp_session_path = os.path.join(lfp_dir, session)
        swr_session_path = os.path.join(swr_dir, f"swrs_session_{session_id}")
        
        if not os.path.exists(swr_session_path):
            print(f"    SWR session path not found: {swr_session_path}")
            continue
        
        folderfiles = os.listdir(swr_session_path)
        global_ripples_filename = [f for f in folderfiles if 'global_swrs' in f]
        if not global_ripples_filename:
            print(f"    No global SWR file found in {swr_session_path}")
            continue
        
        global_ripples_df = pd.read_csv(os.path.join(swr_session_path, global_ripples_filename[0]), compression='gzip')
        
        # Load speed data based on dataset type
        try:
            if dataset_type in ["abi_visbehave", "abi_viscoding"]:
                session_obj = getattr(cache, config["session_loader"])(session_id)
                wheel_velocity, wheel_time = get_speed_data_abi(session_obj, dataset_type)
            else:  # ibl
                wheel_velocity, wheel_time = get_speed_data_ibl(one, session_id)
        except Exception as e:
            print(f"    Error loading speed data for session {session_id}: {e}")
            continue
        
        lfp_files = os.listdir(lfp_session_path)
        for file in folderfiles:
            if not 'putative_swr_events' in file:
                continue
            
            events_df = pd.read_csv(os.path.join(swr_session_path, file), compression='gzip')
            if 'Overlaps_with_gamma' in events_df.columns and 'Overlaps_with_movement' in events_df.columns:
                events_df = events_df[(events_df.Overlaps_with_gamma == True) & (events_df.Overlaps_with_movement == True)]
            
            probe_id = re.search(r"probe_(.*?)_", file).group(1)
            channel_indx = re.search(r"channel_(.*?)_", file).group(1)
            
            lfp_data_file = [f for f in lfp_files if f"channel_{channel_indx}" in f and probe_id in f and "ca1_putative_pyramidal_layer.npz" in f]
            if not lfp_data_file:
                print(f"    No LFP data file for channel {channel_indx} probe {probe_id}")
                continue
            
            lfp_data = np.load(os.path.join(lfp_session_path, lfp_data_file[0]))['lfp_ca1']
            lfp_times_file = [f for f in lfp_files if f"channel_{channel_indx}" in f and probe_id in f and "time_index_1500hz.npz" in f]
            if not lfp_times_file:
                print(f"    No LFP time index file for channel {channel_indx} probe {probe_id}")
                continue
            
            lfp_times = np.load(os.path.join(lfp_session_path, lfp_times_file[0]))['lfp_time_index']
            theta_pow = np.convolve(lfp_data.flatten(), theta_filter.flatten(), mode='same')
            theta_pow_zscore = zscore(np.abs(hilbert(theta_pow)) ** 2)
            
            for _, event in events_df.iterrows():
                event_window_mask = (lfp_times >= event['start_time']) & (lfp_times <= event['end_time'])
                event_theta = theta_pow_zscore[event_window_mask]
                mean_theta = np.mean(event_theta) if event_theta.size > 0 else np.nan
                
                # Compute mean speed in event window
                mask = (wheel_time >= event['start_time']) & (wheel_time <= event['end_time'])
                mean_speed = np.abs(wheel_velocity[mask]).mean() if mask.any() else np.nan
                
                results.append({
                    'session': session,
                    'probe_id': probe_id,
                    'channel_indx': channel_indx,
                    'start_time': event['start_time'],
                    'end_time': event['end_time'],
                    'duration': event['end_time'] - event['start_time'],
                    'mean_theta_power': mean_theta,
                    'mean_speed': mean_speed
                })
    
    out_path = os.path.join(OUTPUT_DIR, f"{dataset_type}_swr_theta_speed.npz")
    np.savez(out_path, results=results)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main() 