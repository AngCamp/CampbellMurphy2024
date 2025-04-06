#!/usr/bin/env python
"""
Script to download Allen LFP data with targeted file cleanup and session-level multiprocessing.
"""

import os
import time
import numpy as np
import boto3
import multiprocessing
from pathlib import Path
from botocore.config import Config
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache
)

# Set cache directory to your existing cache
SDK_CACHE_DIR = '/space/scratch/allen_visbehave_data'
pool_size = 3
MAX_SESSION_ATTEMPTS = 5  # Add a maximum retry count for sessions

# Configure longer timeouts for boto3
my_config = Config(
    connect_timeout=1800,  # 30 minutes
    read_timeout=1800,     # 30 minutes
    retries={'max_attempts': 5, 'mode': 'adaptive'}
)
s3 = boto3.client('s3', config=my_config)

def remove_session_file(session_id):
    """Remove just the main session file, not the entire folder."""
    session_path = os.path.join(
        SDK_CACHE_DIR,
        "visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions",
        str(session_id)
    )
    
    if not os.path.exists(session_path):
        print(f"Session folder does not exist: {session_path}")
        return False
    
    main_file = f"ecephys_session_{session_id}.nwb"
    main_file_path = os.path.join(session_path, main_file)
    
    if os.path.exists(main_file_path):
        try:
            print(f"Removing corrupted session file: {main_file_path}")
            os.remove(main_file_path)
            print("Successfully removed session file")
            return True
        except Exception as e:
            print(f"Error removing session file: {e}")
            return False
    else:
        print(f"Session file not found: {main_file_path}")
        return False

def find_and_remove_probe_lfp_file(session_id: int, probe_id: int, cache) -> bool:
    """
    Delete the LFP file that belongs to `probe_id` inside a session folder.
    Based on directory listing, expects files like 'probe_probeB_lfp.nwb'.

    Parameters
    ----------
    session_id : int
        Ecephys session ID.
    probe_id   : int
        Numeric probe_id (as in the probe table).
    cache      : VisualBehaviorNeuropixelsProjectCache
        A *live* cache instance so we can translate probe_id → probe_label.
    """
    session_path = (
        Path(SDK_CACHE_DIR)
        / "visual-behavior-neuropixels-0.5.0"
        / "behavior_ecephys_sessions"
        / str(session_id)
    )

    if not session_path.exists():
        print(f"[{session_id}] session folder missing: {session_path}")
        return False

    # ── translate numeric probe_id → string label ("probeB", "probeE", …) ──
    try:
        probe_table = cache.get_probe_table()
        probe_label = probe_table.loc[probe_id, "name"]  # e.g. 'probeB'
    except Exception as e:
        print(f"[{session_id}] could not map probe_id {probe_id} → label: {e}")
        return False

    # Based on the directory listing, the correct format is "probe_probeX_lfp.nwb"
    target_file = session_path / f"probe_{probe_label}_lfp.nwb"
    
    if target_file.exists():
        try:
            print(f"Removing corrupted LFP file → {target_file}")
            target_file.unlink()
            return True
        except Exception as e:
            print(f"❌  Failed to delete {target_file}: {e}")
            return False
    else:
        # Log that we didn't find the file in the expected format
        print(f"[{session_id}] LFP file not found at expected path: {target_file}")
        
        # Fall back to searching for any file containing the probe name and "lfp"
        for file in session_path.glob(f"*{probe_label}*lfp*.nwb"):
            try:
                print(f"Found alternative LFP file → {file}")
                file.unlink()
                return True
            except Exception as e:
                print(f"❌  Failed to delete {file}: {e}")
                return False
    
    print(f"[{session_id}] No LFP file found for {probe_label}")
    return False

def process_single_session(session_id):
    """
    Process a single session in a loop. 
    If everything (session + probes) downloads successfully, we break. 
    If it fails in ways that used to cause a 'skip', we wait and retry up to a maximum number of times.
    """
    session_attempts = 0
    
    while session_attempts < MAX_SESSION_ATTEMPTS:  # Prevent infinite loops
        session_attempts += 1
        print(f"\nProcessing session {session_id} (attempt {session_attempts}/{MAX_SESSION_ATTEMPTS})...")
        
        try:
            # Create the cache object fresh each time in this process
            cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
                cache_dir=SDK_CACHE_DIR
            )
            
            # Patch internal S3 client if present
            if hasattr(cache, 'fetch_api') and hasattr(cache.fetch_api, '_cloud_cache'):
                if hasattr(cache.fetch_api._cloud_cache, '_s3_client'):
                    cache.fetch_api._cloud_cache._s3_client._client_config = my_config
            
            # First try to download the session
            try:
                print(f"Downloading session {session_id}")
                session = cache.get_ecephys_session(ecephys_session_id=session_id)
                print(f"Successfully downloaded session {session_id}")
            except Exception as e:
                print(f"Error loading session {session_id}: {str(e)}")
                # Check for known file-corruption/timeouts
                if ("truncated file" in str(e) or 
                    "Unable to synchronously open file" in str(e) or 
                    "Read timeout" in str(e)):
                    print("Detected corrupted session file or timeout, removing main file...")
                    remove_session_file(session_id)
                    print(f"Retrying download of session {session_id}...")
                    try:
                        session = cache.get_ecephys_session(ecephys_session_id=session_id)
                        print(f"Successfully downloaded session {session_id} after cleanup")
                    except Exception as retry_e:
                        print(f"Failed again after cleanup: {str(retry_e)}")
                        # Wait and retry entire session download from scratch
                        time.sleep(60)
                        continue
                else:
                    # For any non-corruption error, we wait and retry
                    print("Non-file-corruption error encountered; waiting 60s and retrying ...")
                    time.sleep(60)
                    continue
            
            # If we got here, session is successfully loaded. Now get the probe table.
            try:
                probes_table = cache.get_probe_table()
                session_probes = probes_table[probes_table.ecephys_session_id == session_id]
                probe_ids = list(session_probes.index)
                print(f"Found {len(probe_ids)} probes for session {session_id}")
            except Exception as e:
                print(f"Error getting probe table for session {session_id}: {str(e)}")
                print("Waiting 60s and retrying this session ...")
                time.sleep(60)
                continue  # Retry the whole session from top

            # Filter for probes that have LFP data
            valid_probes = []
            for pid in probe_ids:
                if pid in session_probes.index and session_probes.loc[pid, 'has_lfp_data']:
                    valid_probes.append(pid)
            
            print(f"Found {len(valid_probes)} probes with LFP data for session {session_id}")
            
            # Track failed probes
            failed_probes = []

            # Process each probe
            for j, probe_id in enumerate(valid_probes):
                print(f"[{j+1}/{len(valid_probes)}] Processing probe {probe_id} for session {session_id}")
                
                # Try up to 3 times for each probe
                success_for_probe = False
                for attempt in range(3):
                    if attempt > 0:
                        print(f"Retry attempt {attempt+1}/3 for probe {probe_id}")
                    
                    try:
                        channels = session.get_channels()
                        ca1_channels = channels[
                            (channels.probe_id == probe_id) & 
                            (channels.structure_acronym == "CA1")
                        ]
                        
                        if len(ca1_channels) == 0:
                            print(f"No CA1 channels found in probe {probe_id}, skipping probe.")
                            success_for_probe = True  # We treat "no data" as not an error
                            break
                        
                        print(f"Found {len(ca1_channels)} CA1 channels in probe {probe_id}")
                        
                        # Download LFP
                        print(f"Downloading LFP for probe {probe_id}")
                        start_time = time.time()
                        lfp = session.get_lfp(probe_id)
                        elapsed = time.time() - start_time
                        print(f"Successfully loaded LFP for probe {probe_id} "
                              f"in {elapsed:.2f} seconds. LFP shape: {lfp.shape}")
                        success_for_probe = True
                        break  # Done with this probe
                        
                    except Exception as e:
                        print(f"Error processing probe {probe_id}: {str(e)}")

                        # Known file corruption scenarios
                        if ("truncated file" in str(e) or 
                            "Unable to synchronously open file" in str(e) or 
                            "Read timeout" in str(e)):
                            print("Detected corrupted LFP file or timeout, removing only this probe's file...")
                            removed = find_and_remove_probe_lfp_file(session_id, probe_id, cache)
                            if not removed:
                                print(f"WARNING: Could not find/remove LFP file for probe {probe_id}!")
                                
                            if attempt < 2:
                                wait_time = (2 ** attempt) * 10  # 10s, 20s
                                print(f"Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                print(f"Failed after 3 attempts for probe {probe_id}.")
                                failed_probes.append(probe_id)
                        else:
                            # Non-corruption error, break attempts for this probe
                            print("Non-corruption error. Stopping attempts for this probe.")
                            failed_probes.append(probe_id)
                            break

                # If the probe never succeeded after all attempts
                if not success_for_probe:
                    failed_probes.append(probe_id)

            # After processing all probes, check if any failed
            if failed_probes:
                print(f"Some probes failed for session {session_id}: {failed_probes}")
                print(f"This was attempt {session_attempts}/{MAX_SESSION_ATTEMPTS} for this session.")
                
                # Only retry if we haven't exceeded the maximum attempts
                if session_attempts < MAX_SESSION_ATTEMPTS:
                    print(f"Will wait 60s and re-attempt the session.")
                    time.sleep(60)
                else:
                    print(f"Reached maximum session attempts ({MAX_SESSION_ATTEMPTS}). Moving on to next session.")
                    break
            else:
                # All probes succeeded
                print(f"All probes processed successfully for session {session_id}.")
                break

        except Exception as e:
            # Catch-all for any unexpected errors. Wait and retry session if attempts remaining.
            print(f"Unexpected error in session {session_id}: {str(e)}.")
            if session_attempts < MAX_SESSION_ATTEMPTS:
                print(f"Retrying after 60s (attempt {session_attempts}/{MAX_SESSION_ATTEMPTS})...")
                time.sleep(60)
            else:
                print(f"Reached maximum session attempts ({MAX_SESSION_ATTEMPTS}). Moving on to next session.")
                break

    # End of while loop
    if session_attempts >= MAX_SESSION_ATTEMPTS:
        print(f"Session {session_id} reached maximum retry attempts. Moving to next session.")
    else:
        print(f"Session {session_id} is fully processed!\n")

def main():
    # Load session IDs
    data_file_path = os.path.join("session_id_lists", "allen_visbehave_ca1_session_ids.npz")
    data = np.load(data_file_path)
    all_sessions = data["data"]
    del data
    
    # Uncomment the line below to override with specific sessions for testing
    all_sessions = [1048189115, 1123100019, 1116941914]
    
    print(f"Loaded {len(all_sessions)} sessions from {data_file_path}")

    # Use a Pool to process sessions in parallel.
    # Adjust processes as desired (e.g., processes=4)
    with multiprocessing.Pool(processes=pool_size) as pool:
        pool.map(process_single_session, all_sessions)

if __name__ == "__main__":
    main()