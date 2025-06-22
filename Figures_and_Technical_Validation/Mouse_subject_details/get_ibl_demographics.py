import argparse
import os
import re
import pandas as pd
from one.api import ONE
from tqdm import tqdm

def get_ibl_demographics(swr_dir, output_file):
    """
    Collects demographic and probe information for IBL sessions.

    Args:
        swr_dir (str): Path to the directory containing SWR processing results.
        output_file (str): Path to save the output CSV file.
    """
    if not os.path.exists(swr_dir):
        print(f"SWR results directory not found at {swr_dir}. Check OUTPUT_DIR.")
        return

    try:
        ONE.setup(base_url="https://openalyx.internationalbrainlab.org", silent=True)
        one = ONE(password="international")
    except Exception as e:
        print(f"Failed to setup ONE API: {e}")
        return

    session_folders = [d for d in os.listdir(swr_dir) if os.path.isdir(os.path.join(swr_dir, d)) and 'session' in d]
    
    demographics = []
    failed_sessions = []
    print("Collecting demographics for IBL sessions...")
    for session_folder in tqdm(session_folders, desc="Processing IBL Sessions"):
        try:
            session_id_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', session_folder)
            if not session_id_match:
                # Fallback for different folder naming conventions
                session_id_match = re.search(r'session_([a-f0-9\-]+)', session_folder)
                if not session_id_match:
                    failed_sessions.append(session_folder)
                    continue

            session_id = session_id_match.group(1)

            folder_path = os.path.join(swr_dir, session_folder)
            folder_files = os.listdir(folder_path)
            num_probes_in_ca1 = sum("karlsson_detector_events" in filename for filename in folder_files)
            
            session_info = one.alyx.rest(f"sessions/{session_id}", "list")
            subject_info = one.alyx.rest("subjects/" + session_info["subject"], "list")
            
            demographics.append({
                'session_id': session_id,
                'subject_id': session_info['subject'],
                'age_weeks': subject_info.get('age_weeks'),
                'sex': subject_info.get('sex'),
                'strain': subject_info.get('strain'),
                'full_genotype': subject_info.get('genotype'),
                'source': subject_info.get('source'),
                'probes_in_ca1': num_probes_in_ca1
            })

        except Exception as e:
            failed_sessions.append((session_folder, str(e)))

    if failed_sessions:
        print("\nCould not process the following sessions:")
        for session, error in failed_sessions:
            print(f"- {session}: {error}")
            
    if not demographics:
        print("No demographic data collected.")
        return

    df = pd.DataFrame(demographics)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved IBL demographics to {output_file}")
    print("\nDataFrame Head:")
    print(df.head())

if __name__ == "__main__":
    output_dir_env = os.environ.get('OUTPUT_DIR')

    if not output_dir_env:
        print("Error: Please set the OUTPUT_DIR environment variable.")
        print("Example: export OUTPUT_DIR=/path/to/results")
    else:
        swr_results_dir = os.path.join(output_dir_env, 'ibl_swr_murphylab2024')
        output_csv_path = "demographics_data/ibl_demographics.csv"
        
        get_ibl_demographics(swr_results_dir, output_csv_path) 