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

# Paths
DATA_ROOT = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
OUTPUT_DIR = "/space/scratch/SWR_final_pipeline/validation_data_figure9"
THETA_FILTER_PATH = "../../Figures_Tables_and_Technical_Validation/Distribution_v2/theta_1500hz_bandpass_filter.npz"

# Dataset folders
DATASETS = {
    "abi_visbehave": {
        "swr": "allen_visbehave_swr_murphylab2024",
        "lfp": "allen_visbehave_swr_murphylab2024_lfp_data"
    },
    "abi_viscoding": {
        "swr": "allen_viscoding_swr_murphylab2024",
        "lfp": "allen_viscoding_swr_murphylab2024_lfp_data"
    },
    "ibl": {
        "swr": "ibl_swr_murphylab2024",
        "lfp": "ibl_swr_murphylab2024_lfp_data"
    }
}

THETA_HALF_WINDOW = 0.125  # seconds

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Gather mean theta power and speed for SWR events.")
    parser.add_argument('--max_sessions', type=int, default=None, help='Limit number of sessions per dataset (for debugging)')
    return parser.parse_args()

def process_abi_visbehave(theta_filter, max_sessions=None):
    dataset = "abi_visbehave"
    print(f"Processing dataset: {dataset}")
    swr_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["swr"])
    lfp_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["lfp"])
    session_dirs = sorted(os.listdir(lfp_dir))
    if max_sessions:
        session_dirs = session_dirs[:max_sessions]
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
        # Find global SWR event file
        global_ripples_filename = [f for f in folderfiles if 'global_swrs' in f]
        if not global_ripples_filename:
            print(f"    No global SWR file found in {swr_session_path}")
            continue
        global_ripples_df = pd.read_csv(os.path.join(swr_session_path, global_ripples_filename[0]), compression='gzip')
        # Load speed data (wheel_velocity, wheel_time) - placeholder: user must adapt to their cache system
        # For now, skip speed calculation (user must fill in)
        wheel_velocity = None
        wheel_time = None
        # Load LFP files for each probe/channel in this session
        lfp_files = os.listdir(lfp_session_path)
        for file in folderfiles:
            if not 'putative_swr_events' in file:
                continue
            events_df = pd.read_csv(os.path.join(swr_session_path, file), compression='gzip')
            # Filter events if needed (e.g., overlaps with gamma/movement)
            if 'Overlaps_with_gamma' in events_df.columns and 'Overlaps_with_movement' in events_df.columns:
                events_df = events_df[(events_df.Overlaps_with_gamma == True) & (events_df.Overlaps_with_movement == True)]
            probe_id = re.search(r"probe_(.*?)_", file).group(1)
            channel_indx = re.search(r"channel_(.*?)_", file).group(1)
            # Load LFP data
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
            # Compute theta power (z-scored)
            theta_pow = np.convolve(lfp_data.flatten(), theta_filter.flatten(), mode='same')
            theta_pow_zscore = zscore(np.abs(hilbert(theta_pow)) ** 2)
            for _, event in events_df.iterrows():
                # Compute mean theta power in event window
                event_window_mask = (lfp_times >= event['start_time']) & (lfp_times <= event['end_time'])
                event_theta = theta_pow_zscore[event_window_mask]
                mean_theta = np.mean(event_theta) if event_theta.size > 0 else np.nan
                # Compute mean speed in event window (placeholder: user must implement speed loading)
                mean_speed = np.nan
                # Save result
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
    # Save results
    out_path = os.path.join(OUTPUT_DIR, f"abi_visbehave_swr_theta_speed.npz")
    np.savez(out_path, results=results)
    print(f"Saved results to {out_path}")

def process_abi_viscoding(theta_filter, max_sessions=None):
    dataset = "abi_viscoding"
    print(f"Processing dataset: {dataset}")
    swr_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["swr"])
    lfp_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["lfp"])
    session_dirs = sorted(os.listdir(lfp_dir))
    if max_sessions:
        session_dirs = session_dirs[:max_sessions]
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
        # Speed data placeholder (user must adapt to their cache system)
        wheel_velocity = None
        wheel_time = None
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
                mean_speed = np.nan  # Placeholder for speed
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
    out_path = os.path.join(OUTPUT_DIR, f"abi_viscoding_swr_theta_speed.npz")
    np.savez(out_path, results=results)
    print(f"Saved results to {out_path}")

def process_ibl(theta_filter, max_sessions=None):
    dataset = "ibl"
    print(f"Processing dataset: {dataset}")
    swr_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["swr"])
    lfp_dir = os.path.join(DATA_ROOT, DATASETS[dataset]["lfp"])
    session_dirs = sorted(os.listdir(lfp_dir))
    if max_sessions:
        session_dirs = session_dirs[:max_sessions]
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
        # Speed data placeholder (user must adapt to their cache system)
        wheel_velocity = None
        wheel_time = None
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
                mean_speed = np.nan  # Placeholder for speed
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
    out_path = os.path.join(OUTPUT_DIR, f"ibl_swr_theta_speed.npz")
    np.savez(out_path, results=results)
    print(f"Saved results to {out_path}")

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    theta_filter = np.load(THETA_FILTER_PATH)
    theta_filter = theta_filter[list(theta_filter.keys())[0]]
    process_abi_visbehave(theta_filter, max_sessions=args.max_sessions)
    process_abi_viscoding(theta_filter, max_sessions=args.max_sessions)
    process_ibl(theta_filter, max_sessions=args.max_sessions)
    # All datasets processed

if __name__ == "__main__":
    main() 