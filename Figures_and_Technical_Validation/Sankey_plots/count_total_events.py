import pandas as pd
import glob
import os

def count_swr_events_by_dataset(base_path):
    """
    Count events that pass through the SWR filtering pipeline (valid SWRs) by dataset.
    Includes both gamma and non-gamma events that meet the SWR criteria.
    """
    datasets = [
        "allen_visbehave_swr_murphylab2024",
        "allen_viscoding_swr_murphylab2024",
        "ibl_swr_murphylab2024"
    ]
    
    dataset_counts = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        session_folders = glob.glob(os.path.join(dataset_path, "swrs_session_*"))
        
        dataset_swr_events = 0
        
        for session in session_folders:
            event_files = glob.glob(os.path.join(session, "*putative_swr_events.csv.gz"))
            
            for event_file in event_files:
                df = pd.read_csv(event_file, compression='gzip')
                
                # Apply the same filtering as the Sankey plot:
                # 1. Sharp wave filter: sw_exceeds_threshold == True AND power_max_zscore > 1
                sharp_wave_mask = (df['sw_exceeds_threshold'] == True) & (df['power_max_zscore'] > 1)
                
                # 2. Movement filter: no overlap with movement
                movement_mask = ~df['overlaps_with_movement']
                
                # Count events that pass both filters (these are valid SWRs)
                valid_swr_mask = sharp_wave_mask & movement_mask
                dataset_swr_events += valid_swr_mask.sum()
        
        dataset_counts[dataset] = dataset_swr_events
    
    return dataset_counts

if __name__ == "__main__":
    # Use the same base path as the Sankey plot script
    base_path = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
    
    try:
        dataset_counts = count_swr_events_by_dataset(base_path)
        
        print("SWR Events by Dataset:")
        print("=" * 50)
        total_events = 0
        for dataset, count in dataset_counts.items():
            print(f"{dataset}: {count:,}")
            total_events += count
        
        print("-" * 50)
        print(f"Total SWR events: {total_events:,}")
        print("(Includes both gamma and non-gamma events that pass SWR filtering criteria)")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Please check that the base path exists and contains the expected dataset structure.") 