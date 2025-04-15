#!/usr/bin/env python
"""
Example script to download Allen VISUAL CODING ecephys data with targeted file cleanup,
using session-level multiprocessing.
"""

import os
import time
import numpy as np
import multiprocessing

# For the visual coding dataset:
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

############################
# CONFIGURATION
############################

# Directory where manifest.json resides AND all cached data will be stored
SDK_CACHE_DIR = "/path/to/your/visual_coding_data"

# Full path to the manifest.json file
MANIFEST_PATH = os.path.join(SDK_CACHE_DIR, "manifest.json")

# Path to your session ID list.  (Replace with your real filename.)
SESSION_ID_LIST_FILE = os.path.join("session_id_lists", "visual_coding_session_ids.npz")

############################
# HELPER FUNCTIONS
############################

def remove_session_file(session_id):
    """
    Remove the primary NWB file for a session if we detect corruption.
    Adjust this path to match how the visual coding dataset
    organizes its NWB files on your disk.
    """
    session_dir = os.path.join(SDK_CACHE_DIR, "sessions", str(session_id))
    if not os.path.exists(session_dir):
        print(f"Session folder does not exist: {session_dir}")
        return False
    
    nwb_filename = f"session_{session_id}.nwb"
    nwb_filepath = os.path.join(session_dir, nwb_filename)

    if os.path.exists(nwb_filepath):
        try:
            print(f"Removing corrupted NWB file: {nwb_filepath}")
            os.remove(nwb_filepath)
            print("Successfully removed session file")
            return True
        except Exception as e:
            print(f"Error removing session file: {e}")
            return False
    else:
        print(f"Session file not found: {nwb_filepath}")
        return False

def find_and_remove_probe_lfp_file(session_id, probe_id):
    """
    Remove a specific probe's LFP file if we detect corruption/timeouts.
    Adjust this path/filename pattern for the visual coding dataset
    if needed. This is an example pattern; adapt to your local structure.
    """
    session_dir = os.path.join(SDK_CACHE_DIR, "sessions", str(session_id))
    if not os.path.exists(session_dir):
        print(f"Session folder does not exist: {session_dir}")
        return False

    # This is just an example name pattern; you may need to adjust.
    # For example, if "probeABC_lfp.nwb" is the naming scheme for your dataset.
    possible_lfps = [
        f for f in os.listdir(session_dir)
        if "probe" in f and "lfp" in f and f.endswith(".nwb")
    ]
    if not possible_lfps:
        print(f"No LFP files found for session {session_id}")
        return False

    # Remove the first matching file or all matching files
    success = False
    for lfp_file in possible_lfps:
        # You could check the probe_id carefully here
        # but for simplicity we'll just remove any found LFP NWB:
        lfp_path = os.path.join(session_dir, lfp_file)
        try:
            print(f"Removing corrupted LFP file: {lfp_path}")
            os.remove(lfp_path)
            print("Successfully removed LFP file")
            success = True
        except Exception as e:
            print(f"Error removing LFP file: {e}")

    return success


############################
# SESSION-LEVEL PROCESSING
############################

def process_single_session(session_id):
    """
    Process a single ecephys session from the VISUAL CODING dataset.
    Retries forever on any fatal error, waiting 60s before each retry.
    """
    while True:
        try:
            print(f"\n=== Processing session {session_id} ===")

            # Build a new cache object on each retry
            cache = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)

            # Try to load the session data
            try:
                print(f"Loading session {session_id} ...")
                session = cache.get_session_data(session_id)
                print(f"Successfully loaded session {session_id}.")
            except Exception as e:
                print(f"Error loading session {session_id}: {e}")
                # Check if it's a corruption/timeout problem
                if ("truncated file" in str(e).lower() or
                    "unable to synchronously open file" in str(e).lower() or
                    "read timeout" in str(e).lower()):
                    print("Detected likely corruption or timeout. Removing main file and retrying...")
                    remove_session_file(session_id)
                    time.sleep(60)
                    continue
                else:
                    # Non-corruption error
                    print("Non-corruption error. Waiting 60s then retrying entire session.")
                    time.sleep(60)
                    continue

            # If we reach here, session is loaded. Let's get the probe IDs.
            # In the Visual Coding dataset, the cache has a `probes` table with
            # "ecephys_session_id". We'll filter that for our session.
            try:
                probes_df = cache.get_probes()
                session_probes = probes_df[probes_df["ecephys_session_id"] == session_id]
                probe_ids = list(session_probes.index)
                print(f"Found {len(probe_ids)} total probes for session {session_id}.")
            except Exception as e:
                print(f"Error getting probes table: {e}")
                print("Waiting 60s, then retrying entire session.")
                time.sleep(60)
                continue

            # We only care about probes that have CA1 channels in this dataset
            # You can adapt the code if you also only want LFP-capable probes, etc.
            probes_of_interest = []
            for pid in probe_ids:
                # Check if CA1 is present among ecephys_structure_acronym
                ch = session.channels[session.channels.probe_id == pid]
                acronyms = ch.ecephys_structure_acronym.unique()
                if "CA1" in acronyms:
                    probes_of_interest.append(pid)

            print(f"Found {len(probes_of_interest)} probes that contain CA1 channels.")

            # Process each probe
            for idx, probe_id in enumerate(probes_of_interest):
                print(f"\n--> Probe {probe_id} ({idx+1}/{len(probes_of_interest)}) in session {session_id}")
                
                success_for_probe = False
                for attempt in range(3):
                    if attempt > 0:
                        print(f"  [Retry {attempt+1}/3] for probe {probe_id}")
                    try:
                        # Attempt to download LFP
                        print("  Getting LFP data...")
                        lfp = session.get_lfp(probe_id)
                        print(f"  LFP loaded. Shape: {lfp.shape}")
                        success_for_probe = True
                        break
                    except Exception as e:
                        print(f"  Error retrieving LFP for probe {probe_id}: {e}")
                        # If it's a corruption or read issue, remove the file and back off
                        err_str = str(e).lower()
                        if ("truncated" in err_str or
                            "unable to synchronously open file" in err_str or
                            "read timeout" in err_str):
                            print("  Detected likely corrupted LFP file. Removing and backing off.")
                            find_and_remove_probe_lfp_file(session_id, probe_id)
                            time.sleep(10 * (2 ** attempt))  # e.g. 10s, 20s, 40s
                        else:
                            # Non-corruption error, skip this probe
                            print("  Non-corruption error, skipping this probe.")
                            break

                # After up to 3 attempts, if still not successful, treat as fatal and retry session
                if not success_for_probe:
                    print(f"\nProbe {probe_id} ultimately failed for session {session_id}.")
                    print("Will wait 60s then retry the entire session.")
                    time.sleep(60)
                    break
            else:
                # If we did NOT break from the probe loop, all probes succeeded or were safely skipped.
                print(f"All relevant probes succeeded for session {session_id}!")
                break

        except Exception as e:
            # Catch-all for unexpected issues at the session level
            print(f"Unexpected session-level error for {session_id}: {e}")
            print("Sleeping 60s, then re-attempting session ...")
            time.sleep(60)
            continue

    # Exiting the while loop means success
    print(f"Session {session_id} is fully processed.\n")


############################
# MAIN MULTIPROCESSING ENTRY
############################

def main():
    # Load session IDs
    data = np.load(SESSION_ID_LIST_FILE)
    all_sessions = data["data"]  # Adjust if your .npz has a different key
    del data
    print(f"Loaded {len(all_sessions)} session IDs from {SESSION_ID_LIST_FILE}")

    # Start a multiprocessing pool to handle sessions in parallel
    n_processes = 4  # Adjust as appropriate
    with multiprocessing.Pool(processes=n_processes) as pool:
        pool.map(process_single_session, all_sessions)

if __name__ == "__main__":
    main()
