import argparse
import os
import re
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from tqdm import tqdm

def get_visbehaviour_subject_info(swr_dir, sdk_dir, output_file):
    """
    Collects subject information for Allen Visual Behaviour sessions.

    Args:
        swr_dir (str): Path to the directory containing SWR processing results.
        sdk_dir (str): Path to the Allen SDK cache directory.
        output_file (str): Path to save the output CSV file.
    """
    if not os.path.exists(sdk_dir):
        print(f"SDK cache directory not found at {sdk_dir}")
        return

    if not os.path.exists(swr_dir):
        print(f"SWR results directory not found at {swr_dir}")
        return

    try:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=sdk_dir)
        sessions_table = cache.get_ecephys_session_table()
    except Exception as e:
        print(f"Failed to load Allen SDK cache: {e}")
        return

    session_dirs = [d for d in os.listdir(swr_dir) if d.startswith("swrs_session_")]
    session_ids = [int(re.sub("swrs_session_", "", d)) for d in session_dirs]

    subject_info_list = []
    print("Collecting subject info for Allen Visual Behaviour sessions...")
    for session_id in tqdm(session_ids, desc="Processing Visual Behaviour Sessions"):
        try:
            # The table is already indexed by ecephys_session_id. No need to set index.
            session_info = sessions_table.loc[session_id]
            session_dir_name = f"swrs_session_{session_id}"
            session_path = os.path.join(swr_dir, session_dir_name)
            
            if not os.path.isdir(session_path):
                continue

            probe_files = [f for f in os.listdir(session_path) if 'putative_swr_events' in f]
            num_probes_in_ca1 = len(probe_files)

            subject_info_list.append({
                'session_id': session_id,
                'specimen_id': session_info['mouse_id'],
                'age_in_days': session_info['age_in_days'],
                'sex': session_info['sex'],
                'full_genotype': session_info['genotype'],
                'probes_in_ca1': num_probes_in_ca1
            })
        except KeyError:
            print(f"Session ID {session_id} not found in the session table. Skipping.")
        except Exception as e:
            print(f"Could not process session {session_id}. Error: {e}")
    
    if not subject_info_list:
        print("No subject information collected.")
        return

    df = pd.DataFrame(subject_info_list)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved visual behaviour subject info to {output_file}")
    print("\nDataFrame Head:")
    print(df.head())

if __name__ == "__main__":
    
    output_dir_env = os.environ.get('OUTPUT_DIR')
    sdk_cache_env = os.environ.get('ABI_VISUAL_BEHAVIOUR_SDK_CACHE')

    if not all([output_dir_env, sdk_cache_env]):
        print("Error: Please set the OUTPUT_DIR and ABI_VISUAL_BEHAVIOUR_SDK_CACHE environment variables.")
        print("Example: export OUTPUT_DIR=/path/to/results")
        print("         export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=/path/to/sdk_cache")
    else:
        swr_results_dir = os.path.join(output_dir_env, 'allen_visbehave_swr_murphylab2024')
        output_csv_path = "subject_info_data/visbehaviour_subject_info.csv"
        
        get_visbehaviour_subject_info(swr_results_dir, sdk_cache_env, output_csv_path) 