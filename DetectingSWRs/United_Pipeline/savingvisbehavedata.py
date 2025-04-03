#!/usr/bin/env python
"""
Simplified script to download Allen LFP data with targeted file cleanup.
"""

import os
import sys
import time
import subprocess
import numpy as np
from botocore.config import Config
import boto3
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache
)

# Set cache directory to your existing cache
SDK_CACHE_DIR = '/space/scratch/allen_visbehave_data'

# Configure longer timeouts for boto3
my_config = Config(
    connect_timeout=1800,  # 30 minutes
    read_timeout=1800,     # 30 minutes
    retries={'max_attempts': 5, 'mode': 'adaptive'}
)
s3 = boto3.client('s3', config=my_config)

def remove_session_file(session_id):
    """Remove just the main session file, not the entire folder"""
    session_path = os.path.join(SDK_CACHE_DIR, "visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions", str(session_id))
    
    if not os.path.exists(session_path):
        print(f"Session folder does not exist: {session_path}")
        return False
    
    # Look for the main session file
    main_file = f"ecephys_session_{session_id}.nwb"
    main_file_path = os.path.join(session_path, main_file)
    
    if os.path.exists(main_file_path):
        try:
            print(f"Removing corrupted session file: {main_file_path}")
            os.remove(main_file_path)
            print(f"Successfully removed session file")
            return True
        except Exception as e:
            print(f"Error removing session file: {e}")
            return False
    else:
        print(f"Session file not found: {main_file_path}")
        return False

def find_and_remove_probe_lfp_file(session_id, probe_id):
    """Find and remove a specific probe's LFP file"""
    session_path = os.path.join(SDK_CACHE_DIR, "visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions", str(session_id))
    
    if not os.path.exists(session_path):
        print(f"Session folder does not exist: {session_path}")
        return False
    
    # The probe files follow a pattern like "probe_probeB_lfp.nwb"
    # Look for files matching any probe pattern
    for filename in os.listdir(session_path):
        if filename.startswith("probe_probe") and filename.endswith("_lfp.nwb"):
            # Check if this is the probe file we want to remove
            # We'd need more sophisticated matching based on probe_id
            # For now, we'll remove any LFP file we find during error handling
            file_path = os.path.join(session_path, filename)
            try:
                print(f"Removing corrupted LFP file: {file_path}")
                os.remove(file_path)
                print(f"Successfully removed LFP file")
                return True
            except Exception as e:
                print(f"Error removing LFP file: {e}")
                return False
    
    print(f"No LFP files found for session {session_id}")
    return False

def main():
    # Load session IDs
    data_file_path = os.path.join("session_id_lists", "allen_visbehave_ca1_session_ids.npz")
    data = np.load(data_file_path)
    all_sessions = data["data"]
    del data
    print(f"Loaded {len(all_sessions)} sessions from {data_file_path}")
    
    # Create cache object using existing cache directory
    print(f"Creating cache in {SDK_CACHE_DIR}")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=SDK_CACHE_DIR)
    
    # Try to patch the internal S3 client with our timeout settings
    if hasattr(cache, 'fetch_api') and hasattr(cache.fetch_api, '_cloud_cache'):
        if hasattr(cache.fetch_api._cloud_cache, '_s3_client'):
            print("Patching internal S3 client with longer timeouts")
            cache.fetch_api._cloud_cache._s3_client._client_config = my_config
    
    # Process each session
    for i, session_id in enumerate(all_sessions):
        print(f"\n[{i+1}/{len(all_sessions)}] Processing session {session_id}")
        
        # Try to download the session
        try:
            print(f"Downloading session {session_id}")
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            print(f"Successfully downloaded session {session_id}")
        except Exception as e:
            print(f"Error loading session {session_id}: {str(e)}")
            
            if "truncated file" in str(e) or "Unable to synchronously open file" in str(e) or "Read timeout" in str(e):
                print(f"Detected corrupted session file or timeout, removing only the main session file...")
                remove_session_file(session_id)
                print(f"Retrying download of session {session_id}...")
                try:
                    session = cache.get_ecephys_session(ecephys_session_id=session_id)
                    print(f"Successfully downloaded session {session_id} after cleanup")
                except Exception as retry_e:
                    print(f"Failed to download session {session_id} even after cleanup: {str(retry_e)}")
                    continue
            else:
                print(f"Skipping session {session_id} due to non-file-corruption error")
                continue
        
        # Get the probe IDs using the probe table
        try:
            probes_table = cache.get_probe_table()
            session_probes = probes_table[probes_table.ecephys_session_id == session_id]
            probe_ids = list(session_probes.index)
            print(f"Found {len(probe_ids)} probes for session {session_id}")
        except Exception as e:
            print(f"Error getting probe table: {str(e)}")
            continue  # Skip to next session
        
        # Filter for probes with LFP data
        valid_probes = []
        for probe_id in probe_ids:
            if probe_id in session_probes.index and session_probes.loc[probe_id, 'has_lfp_data']:
                valid_probes.append(probe_id)
        
        print(f"Found {len(valid_probes)} probes with LFP data")
        
        # Process each probe
        for j, probe_id in enumerate(valid_probes):
            print(f"[{j+1}/{len(valid_probes)}] Processing probe {probe_id}")
            
            # Try up to 3 times to process this probe
            for attempt in range(3):
                if attempt > 0:
                    print(f"Retry attempt {attempt+1}/3 for probe {probe_id}")
                
                try:
                    # Get channels for this probe
                    channels = session.get_channels()
                    ca1_channels = channels[
                        (channels.probe_id == probe_id) & 
                        (channels.structure_acronym == "CA1")
                    ]
                    
                    if len(ca1_channels) == 0:
                        print(f"No CA1 channels found in probe {probe_id}, skipping")
                        break  # Move to next probe
                    
                    print(f"Found {len(ca1_channels)} CA1 channels in probe {probe_id}")
                    
                    # Download LFP for this probe
                    print(f"Downloading LFP for probe {probe_id}")
                    start_time = time.time()
                    lfp = session.get_lfp(probe_id)
                    print(f"Successfully loaded LFP in {time.time() - start_time:.2f} seconds")
                    print(f"LFP data shape: {lfp.shape}")
                    break  # Success, move to next probe
                    
                except Exception as e:
                    print(f"Error processing probe {probe_id}: {str(e)}")
                    
                    if "truncated file" in str(e) or "Unable to synchronously open file" in str(e) or "Read timeout" in str(e):
                        # Only remove the specific probe's LFP file
                        print(f"Detected corrupted LFP file or timeout, removing only this probe's file...")
                        find_and_remove_probe_lfp_file(session_id, probe_id)
                        
                        if attempt < 2:  # 0-indexed, so 0, 1 are first two attempts
                            wait_time = (2 ** attempt) * 10  # 10s, 20s
                            print(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to download LFP for probe {probe_id} after 3 attempts")
                    else:
                        # Non-corruption error, skip this probe
                        print(f"Skipping probe {probe_id} due to non-corruption error")
                        break

if __name__ == "__main__":
    main()