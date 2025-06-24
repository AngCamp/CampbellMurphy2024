import os
import re
import pandas as pd
from one.api import ONE
from tqdm import tqdm
import numpy as np

def get_ibl_subject_info(swr_dir, output_file):
    """
    Collects subject information and probe counts for IBL sessions.

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
    
    subject_info_list = []
    failed_sessions = []
    print("Collecting subject info for IBL sessions...")
    for session_folder in tqdm(session_folders, desc="Processing IBL Sessions"):
        try:
            session_id_match = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', session_folder)
            if not session_id_match:
                session_id_match = re.search(r'session_([a-f0-9\-]+)', session_folder)
                if not session_id_match:
                    failed_sessions.append(session_folder)
                    continue

            session_id = session_id_match.group(1)

            folder_path = os.path.join(swr_dir, session_folder)
            folder_files = os.listdir(folder_path)
            num_probes_in_ca1 = sum("putative_swr_events" in filename for filename in folder_files)
            
            session_info = one.alyx.rest(f"sessions/{session_id}", "list")
            subject_details = one.alyx.rest("subjects/" + session_info["subject"], "list")
            
            subject_info_list.append({
                'session_id': session_id,
                'subject_id': session_info['subject'],
                'age_weeks': subject_details.get('age_weeks'),
                'sex': subject_details.get('sex'),
                'strain': subject_details.get('strain'),
                'full_genotype': subject_details.get('genotype'),
                'source': subject_details.get('source'),
                'probes_in_ca1': num_probes_in_ca1
            })

        except Exception as e:
            failed_sessions.append((session_folder, str(e)))
            
    if not subject_info_list:
        print("No subject information collected.")
        return

    temp_df = pd.DataFrame(subject_info_list)

    # --- Create Detailed Data Quality Report ---
    report_lines = []
    report_lines.append("--- IBL Data Quality Report ---")
    report_lines.append(f"Found {len(session_folders)} session folders.")
    report_lines.append(f"Successfully processed info for {len(subject_info_list)} sessions.")
    if failed_sessions:
        report_lines.append(f"Failed to process {len(failed_sessions)} sessions/folders.")

    unique_subjects_total = temp_df['subject_id'].nunique()
    report_lines.append(f"Found {unique_subjects_total} unique subjects across all processed sessions.")
    
    sessions_per_subject = temp_df.groupby('subject_id')['session_id'].nunique()
    subjects_with_multiple_sessions = sessions_per_subject[sessions_per_subject > 1]
    if not subjects_with_multiple_sessions.empty:
        report_lines.append(f"\nNote: {len(subjects_with_multiple_sessions)} subjects have more than one session.")
        report_lines.append("The final summary counts unique subjects, not sessions.")

    report_lines.append(f"\n--- Missing Data Report ---")
    
    missing_sex_df = temp_df[temp_df['sex'].isna()]
    subjects_without_sex = missing_sex_df['subject_id'].unique()
    if len(subjects_without_sex) > 0:
        sessions_of_imputed_subjects = missing_sex_df['session_id'].unique()
        report_lines.append(f"Found {len(subjects_without_sex)} subjects with missing 'sex' information.")
        report_lines.append("For the summary, these subjects will be assumed to be 'Male'.")
        
        missing_sex_output_path = os.path.join('subject_info_data', 'ibl_missing_sex_session_ids.npz')
        os.makedirs(os.path.dirname(missing_sex_output_path), exist_ok=True)
        np.savez(missing_sex_output_path, session_ids=sessions_of_imputed_subjects)
        report_lines.append(f"A list of their session IDs has been saved to: {missing_sex_output_path}")
    else:
        report_lines.append("No subjects found with missing 'sex' information.")

    missing_age_df = temp_df[temp_df['age_weeks'].isna()]
    subjects_without_age = missing_age_df['subject_id'].unique()
    if len(subjects_without_age) > 0:
        report_lines.append(f"Found {len(subjects_without_age)} subjects with missing 'age_weeks' information. These subjects will be excluded from age calculations in the summary.")
    else:
        report_lines.append("No subjects found with missing 'age_weeks' information.")
        
    report_lines.append("\n---------------------------------")
    
    report_string = "\n".join(report_lines)
    print(report_string)
    report_path = os.path.join('subject_info_data', 'ibl_data_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_string)
    print(f"Full data quality report saved to: {report_path}")
    # --- End Report ---

    # Impute missing sex as 'M' and save
    df = temp_df.fillna({'sex': 'M'})
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved IBL subject information to {output_file}")
    print("\nDataFrame Head:")
    print(df.head())

if __name__ == "__main__":
    output_dir_env = os.environ.get('OUTPUT_DIR')

    if not output_dir_env:
        print("Error: Please set the OUTPUT_DIR environment variable.")
        print("Example: export OUTPUT_DIR=/path/to/results")
    else:
        swr_results_dir = os.path.join(output_dir_env, 'ibl_swr_murphylab2024')
        output_csv_path = "subject_info_data/ibl_subject_info.csv"
        
        get_ibl_subject_info(swr_results_dir, output_csv_path) 