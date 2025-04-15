#!/usr/bin/env python
"""
Script to download Allen LFP data with targeted file cleanup and session-level multiprocessing.
No retry limits - will continue until all CA1 probes are loaded or fail with non-timeout errors.
"""

import os
import time
import json
import numpy as np
import boto3
import multiprocessing
from pathlib import Path
from botocore.config import Config
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache
)
from multiprocessing import Manager

# Set cache directory to your existing cache
SDK_CACHE_DIR = '/space/scratch/allen_visbehave_data'
pool_size = 10

# Configure longer timeouts for boto3
my_config = Config(
    connect_timeout=1800,  # 30 minutes
    read_timeout=1800      # 30 minutes
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

def process_single_session(session_id, error_log):
    """
    Process a single session until:
    1. All probes with CA1 channels are successfully loaded, or
    2. The only failures are non-timeout/non-corruption errors
    
    Parameters
    ----------
    session_id : int
        Ecephys session ID.
    error_log : multiprocessing.managers.DictProxy
        Shared dictionary to track non-timeout errors.
    """
    # Track which probes we've successfully processed
    successful_probes = set()
    # Track which probes failed for other reasons (these won't be retried)
    other_error_probes = set()
    # Track all probes with CA1 channels
    ca1_probes = set()
    
    session_attempt = 0
    
    # Continue until all CA1 probes are successful or have non-timeout errors
    while True:
        session_attempt += 1
        # Track which probes had timeout/corruption errors in this attempt
        current_timeout_probes = set()
        
        print(f"\nProcessing session {session_id} (attempt {session_attempt})...")
        
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
            
            # Filter for probes that have LFP data and CA1 channels
            # We'll only process probes that aren't already successful
            probes_to_process = []
            ca1_probes_this_attempt = set()  # Track CA1 probes found in this attempt
            
            for pid in valid_probes:
                # Skip already successful probes
                if pid in successful_probes:
                    print(f"Skipping already successful probe {pid}")
                    continue
                    
                # Skip probes with non-timeout errors (we don't retry these)
                if pid in other_error_probes:
                    print(f"Skipping probe {pid} with non-timeout error")
                    continue
                
                # Check if this probe has CA1 channels
                try:
                    channels = session.get_channels()
                    ca1_channels = channels[
                        (channels.probe_id == pid) & 
                        (channels.structure_acronym == "CA1")
                    ]
                    
                    if len(ca1_channels) > 0:
                        print(f"Found {len(ca1_channels)} CA1 channels in probe {pid}")
                        ca1_probes_this_attempt.add(pid)
                        probes_to_process.append(pid)
                        ca1_probes.add(pid)
                    else:
                        print(f"No CA1 channels found in probe {pid}, skipping")
                except Exception as e:
                    print(f"Error checking CA1 channels for probe {pid}: {str(e)}")
                    # If we can't even check, we'll still try to process it
                    probes_to_process.append(pid)
            
            print(f"Found {len(ca1_probes)} probes with CA1 channels for session {session_id}")
            print(f"Will process {len(probes_to_process)} probes in this attempt")
            
            # Process each probe that needs processing
            failed_probes = []
            
            for j, probe_id in enumerate(probes_to_process):
                print(f"[{j+1}/{len(probes_to_process)}] Processing probe {probe_id} for session {session_id}")
                
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
                        error_msg = str(e)
                        print(f"Error processing probe {probe_id}: {error_msg}")
                        
                        # Known file corruption scenarios
                        if ("truncated file" in error_msg or 
                            "Unable to synchronously open file" in error_msg or 
                            "Read timeout" in error_msg):
                            print("Detected corrupted LFP file or timeout, removing only this probe's file...")
                            removed = find_and_remove_probe_lfp_file(session_id, probe_id, cache)
                            if not removed:
                                print(f"WARNING: Could not find/remove LFP file for probe {probe_id}!")
                            
                            # Track this probe as having timeout/corruption issues in this attempt
                            current_timeout_probes.add(probe_id)
                                
                            if attempt < 2:
                                wait_time = (2 ** attempt) * 10  # 10s, 20s
                                print(f"Waiting {wait_time} seconds before retry...")
                                time.sleep(wait_time)
                            else:
                                print(f"Failed after 3 attempts for probe {probe_id}.")
                                failed_probes.append(probe_id)
                        else:
                            # Non-corruption error, log it and stop attempts for this probe
                            print("Non-corruption error. Stopping attempts for this probe.")
                            # Get probe name for the error log
                            probe_name = "unknown"
                            try:
                                probe_name = cache.get_probe_table().loc[probe_id, "name"]
                            except:
                                pass
                                
                            # Add to error log with unique key
                            error_key = f"{session_id}_{probe_id}"
                            error_log[error_key] = {
                                "session_id": session_id,
                                "probe_id": probe_id,
                                "probe_name": probe_name,
                                "error": error_msg
                            }
                            
                            # Mark as other error type
                            other_error_probes.add(probe_id)
                            failed_probes.append(probe_id)
                            break
                
                # If the probe succeeded, add it to successful probes
                if success_for_probe:
                    successful_probes.add(probe_id)
            
            # Evaluate if we need to continue processing this session
            # Calculate our progress
            total_ca1_probes = len(ca1_probes)
            successful_ca1_probes = len([p for p in successful_probes if p in ca1_probes])
            error_ca1_probes = len([p for p in other_error_probes if p in ca1_probes])
            timeout_ca1_probes = len([p for p in current_timeout_probes if p in ca1_probes])
            
            print(f"\nSession {session_id} progress (attempt {session_attempt}):")
            print(f"  - Total CA1 probes: {total_ca1_probes}")
            print(f"  - Successfully processed: {successful_ca1_probes}")
            print(f"  - Non-timeout errors: {error_ca1_probes}")
            print(f"  - Timeout errors in this attempt: {timeout_ca1_probes}")
            
            # Check if all CA1 probes are either successful or have non-timeout errors
            if successful_ca1_probes + error_ca1_probes == total_ca1_probes:
                print(f"All CA1 probes have been successfully processed or have non-timeout errors.")
                print(f"Session {session_id} complete!")
                break
            
            # If there are timeout errors, we need to retry those probes
            if timeout_ca1_probes > 0:
                print(f"Probes with timeout/corruption errors: {list(current_timeout_probes)}")
                print(f"Will wait 60s and re-attempt these probes.")
                time.sleep(60)
                continue
        
        except Exception as e:
            # Catch-all for any unexpected errors. Wait and retry session.
            print(f"Unexpected error in session {session_id}: {str(e)}.")
            print(f"Retrying after 60s...")
            time.sleep(60)
            continue
    
    # End of while loop - Final status report
    successful_ca1_count = len([p for p in successful_probes if p in ca1_probes])
    other_error_ca1_count = len([p for p in other_error_probes if p in ca1_probes])
    
    print(f"\nFinal status for session {session_id}:")
    print(f"  - Successfully processed {successful_ca1_count} out of {len(ca1_probes)} CA1 probes")
    
    if successful_ca1_count == len(ca1_probes):
        print(f"✅ Session {session_id} is fully processed! All CA1 probes loaded successfully.")
    else:
        print(f"✓ Session {session_id} processed all possible probes.")
        print(f"  - {other_error_ca1_count} probes had non-timeout errors (logged)")

def main():
    # Create a manager to share data between processes
    manager = Manager()
    # Create a shared dictionary to track errors
    error_log = manager.dict()
    
    # Load session IDs
    data_file_path = os.path.join("session_id_lists", "allen_visbehave_ca1_session_ids.npz")
    data = np.load(data_file_path)
    all_sessions = data["data"]
    del data
    
    # Uncomment the line below to override with specific sessions for testing
    # all_sessions = [1048189115, 1123100019, 1116941914]
    
    print(f"Loaded {len(all_sessions)} sessions from {data_file_path}")

    # Pass the session IDs and shared error log to the processing function
    args = [(session_id, error_log) for session_id in all_sessions]
    
    # Use a Pool to process sessions in parallel
    with multiprocessing.Pool(processes=pool_size) as pool:
        pool.starmap(process_single_session, args)
    
    # Save error log to a JSON file after all processing is complete
    error_log_dict = dict(error_log)
    with open("error_log.json", "w") as f:
        json.dump(error_log_dict, f, indent=2)
    
    print(f"Error log saved to error_log.json")
    print(f"Total errors logged: {len(error_log_dict)}")

if __name__ == "__main__":
    main()