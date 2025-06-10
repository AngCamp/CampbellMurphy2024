#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
from SWRExplorer import SWRExplorer
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1" # prevent pychace from being written

# Path to sharp wave filter(s)
# Main filter used for Allen/IBL Neuropixels LFP (1500 Hz):
filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/SWR_Neuropixels_Detector/Filters/sharpwave_componenet_8to40band_1500hz_band.npz"
envelope_mode = 'zscore'  # Options: 'zscore' or 'raw'
info_on = False
file_type = 'svg'
output_dir = "top_swr_events/v2_version"

def find_top_events(explorer, min_sw_power=1.5, min_duration=0.08, max_duration=0.1,
                   min_clcorr=0.8, max_speed=5.0, window=0.5, target_count=10):
    """
    Find the top events across all datasets that meet the specified criteria, filtering by speed only for the top events as needed.
    """
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir="/space/scratch/allen_visbehave_data"
    )
    
    all_candidates = []
    # 1. Gather all candidate events (no speed filtering)
    for dataset in ["allen_visbehave_swr_murphylab2024"]:
        print(f"\nProcessing dataset: {dataset}")
        if dataset not in explorer.data:
            continue
        for session_id in explorer.data[dataset]:
            for probe_id in explorer.data[dataset][session_id]:
                best_events = explorer.find_best_events(
                    dataset=dataset,
                    session_id=session_id,
                    probe_id=probe_id,
                    min_sw_power=min_sw_power,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_clcorr=min_clcorr,
                    exclude_gamma=True,
                    exclude_movement=True
                )
                if len(best_events) > 0:
                    if 'overlaps_with_gamma' not in best_events.columns or 'overlaps_with_movement' not in best_events.columns:
                        print(f"[ERROR] Missing expected columns in events for dataset={dataset}, session={session_id}, probe={probe_id}")
                        print(f"Columns found: {list(best_events.columns)}")
                        print(f"File may be corrupted or from an old pipeline run.")
                        raise KeyError("Missing 'overlaps_with_gamma' or 'overlaps_with_movement' in event file!")
                    print(f"[DEBUG] Loaded events for dataset={dataset}, session={session_id}, probe={probe_id}, columns={list(best_events.columns)}, shape={best_events.shape}")
                    # Exclude events with power_max_zscore > 6
                    #if 'power_max_zscore' in best_events.columns:
                    #    powerfilt = (best_events['power_max_zscore'] <= 10) & (best_events['power_max_zscore'] >= 5)
                    #    best_events = best_events[powerfilt]
                    all_candidates.append(best_events)
    if len(all_candidates) == 0:
        print("No events found matching criteria.")
        return pd.DataFrame()
    # 2. Sort all events by sw_ripple_clcorr
    all_candidates_df = pd.concat(all_candidates)
    sorted_events = all_candidates_df.sort_values('sw_ripple_clcorr', ascending=False)
    
    # 3. Iterate through sorted events, filter by speed as needed
    speed_cache = {}  # session_id -> running_speed_df
    selected_events = []
    for event_id, event in sorted_events.iterrows():
        session_id = event['session_id']
        dataset = event['dataset']
        if session_id not in speed_cache:
            try:
                session = cache.get_ecephys_session(int(session_id))
                speed_cache[session_id] = session.running_speed
            except Exception as e:
                print(f"    Could not load session {session_id}: {e}")
                continue
        running_speed_df = speed_cache[session_id]
        # Compute speed for this event
        speeds = explorer.get_allensdk_speed(
            running_speed_df=running_speed_df,
            events_df=pd.DataFrame([event]),
            window=window,
            agg='max'
        )
        max_speed_in_window = float(speeds.iloc[0])
        if max_speed_in_window <= max_speed:
            event = event.copy()
            event['max_speed_in_window'] = max_speed_in_window
            selected_events.append(event)
        if len(selected_events) >= target_count:
            break
    if len(selected_events) == 0:
        print("No events passed the speed filter.")
        return pd.DataFrame()
    return pd.DataFrame(selected_events)

def plot_and_save_events(explorer, events_df, output_dir):
    """
    Plot and save the specified events.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, (event_id, event) in enumerate(events_df.iterrows()):
        session_events = explorer.find_best_events(
            dataset=event['dataset'],
            session_id=event['session_id'],
            probe_id=event['probe_id'],
            min_sw_power=0,
            min_duration=0,
            max_duration=float('inf'),
            min_clcorr=0
        )
        # Do not extract or pass channel IDs; let SWRExplorer handle file selection
        explorer.plot_swr_event(
            events_df=session_events,
            event_idx=event_id,
            filter_path=filter_path,
            envelope_mode=envelope_mode,
            panels_to_plot=[
                'raw_pyramidal_lfp',
                'raw_s_radiatum_lfp',
                'bandpass_signals',
                'envelope',
                'power',
            ],
            show_info_title=info_on,
            show_peak_dots=True
        )
        plt.savefig(f"{output_dir}/event_{idx+1}_{event.index+1}_session_{event['session_id']}_probe_{event['probe_id']}.{file_type}")
        plt.close()

def main():
    # Initialize the explorer with explicit base path
    base_path = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data"
    explorer = SWRExplorer(base_path=base_path)
    
    # Find top events
    top_events = find_top_events(
        explorer=explorer,
        min_sw_power=1.5,
        min_duration=0.08,
        max_duration=0.1,
        min_clcorr=0.8,
        max_speed=5.0,
        window=0.5,
        target_count=10
    )
    
    # Print the top events
    print("\nTop 10 Events:")
    for idx, (event_id, event) in enumerate(top_events.iterrows()):
        print(f"Event {idx+1}:")
        print(f"  Dataset: {event['dataset']}")
        print(f"  Session: {event['session_id']}")
        print(f"  Probe: {event['probe_id']}")
        print(f"  Event ID: {event_id}")
        print(f"  Sharp Wave Power: {event['sw_peak_power']:.2f}")
        print(f"  Correlation: {event['sw_ripple_clcorr']:.2f}")
        print(f"  Max Speed: {event['max_speed_in_window']:.2f} cm/s")
        print()
    
    # Plot and save the events
    plot_and_save_events(
        explorer=explorer,
        events_df=top_events,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main() 