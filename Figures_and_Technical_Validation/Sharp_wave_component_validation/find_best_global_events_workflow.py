#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from SWRExplorer import SWRExplorer
import os
import ast
from scipy.signal import fftconvolve, butter, filtfilt, hilbert, convolve
os.environ["PYTHONDONTWRITEBYTECODE"] = "1" # prevent pycache from being written

# Path to sharp wave filter(s)
filter_path = "/home/acampbell/NeuropixelsLFPOnRamp/SWR_Neuropixels_Detector/Filters/sharpwave_componenet_8to40band_1500hz_band.npz"
output_dir = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_and_Technical_Validation/Sharp_wave_component_validation/top_global_swr_events"
file_ext = 'png'  # User can set this to 'svg', 'pdf', etc.
window = 0.15  # seconds around middle-most peak

# Set this to True to display AP coordinate in y-labels of plots
show_ap_in_ylabel = True

# Direction selection: 'anterior', 'posterior', 'directional' (either), or 'non_directional'
direction_filter = 'directional'

# Option for what to plot as the additional value in global event plots: 'ripple_band', 'envelope', or 'power'
#   'ripple_band': z-scored ripple band (no smoothing)
#   'envelope': z-scored, smoothed ripple envelope (abs(Hilbert), Gaussian smooth, z-score)
#   'power': z-scored, smoothed ripple envelope squared
additional_value = 'power'  # 'ripple_band' or 'envelope' or 'power'

# Duration filter for global events (seconds)
max_duration = 0.15  # Only include events with duration <= max_duration

# Option to plot CSD slices for each global event
plot_csd_slices = False

# Debug flag for printing per-event directionality details
debug_directional_details = False  # Set to True to print per-event direction assignment

# Number of top global events to find and plot
# Set to any integer (e.g., 1, 2, 5, 10, etc.)
top_n_events = 10

# Only include sessions with a number of probes in this range (inclusive)
min_probes = 3
max_probes = 4
min_global_peak_power = 5
max_global_peak_power = 10
speed_threshold = 2.0

# Minimum delay (in seconds) between event peaks for directionality
min_peak_delay = 0.01  # User can set this value

# Toggle collection and printing of directional event summary statistics
collect_directional_stats = True  # Set to False to skip summary stats

# Option to plot time relative to the global peak (True) or in absolute time (False)
relative_time = True  # Set to False for absolute time axis

def contiguity_score(mask):
    # mask: list of bools
    max_block = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_block = max(max_block, current)
        else:
            current = 0
    return max_block

def find_top_global_events(explorer, dataset="allen_visbehave_swr_murphylab2024", min_probe_count=2, min_global_peak_power=5.0, max_global_peak_power=10.0, target_count=10, min_peak_delay=0.002):
    """
    Find the top global events by global_peak_power and probe count, only considering events where the peak event is present in the probe events.
    Only include events with global_peak_power between 5 and 10.
    Prioritize events with high spatial contiguity of participating probes.
    Filter events by direction based on direction_filter parameter.
    Filter events by duration (<= max_duration) after directionality filtering.
    Only include sessions with a number of probes in [min_probes, max_probes].
    Directionality is only assigned if ALL peak times are strictly increasing or strictly decreasing with at least min_peak_delay between each.
    """
    all_candidates = []
    for session_dir in (explorer.base_path / dataset).glob("swrs_session_*"):
        # Count number of putative_swr_events files (probes) in this session
        n_probes = len(list(session_dir.glob("probe_*_putative_swr_events.csv.gz")))
        if n_probes < min_probes or n_probes > max_probes:
            continue  # Skip this session
        session_id = session_dir.name.split('_')[-1]
        global_csvs = list(session_dir.glob(f"session_{session_id}_global_swr_events.csv.gz"))
        if not global_csvs:
            continue
        global_events = pd.read_csv(global_csvs[0], index_col=0)
        # --- Compute direction for all events FIRST ---
        # Get AP coordinates for all probes in this session
        lfp_dir = explorer.base_path / explorer.lfp_sources[dataset] / f"lfp_session_{session_id}"
        probe_ap_dict = {}
        for lfp_file in lfp_dir.glob("probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"):
            m_probe = lfp_file.name.split('_')[1]
            m_chan = lfp_file.name.split('_')[3]
            ap = explorer.get_channel_ap_coordinate(int(m_chan)) if m_chan.isdigit() else None
            probe_ap_dict[m_probe] = ap
        # Compute direction for each event
        valid_events = []
        directions = []
        keep_mask = []
        for idx, row in global_events.iterrows():
            participating_probes = ast.literal_eval(row['participating_probes'])
            peak_times = ast.literal_eval(row['peak_times'])
            peak_powers = ast.literal_eval(row['peak_powers'])
            # Convert all probe IDs to strings for consistent comparison
            participating_probes = [str(pid) for pid in participating_probes]
            participating_probes_ap = [(pid, probe_ap_dict.get(pid, float('inf'))) for pid in participating_probes]
            participating_probes_ap.sort(key=lambda x: x[1])
            sorted_probe_ids = [p[0] for p in participating_probes_ap]
            # Add error checking and debug info
            print(f"[DEBUG] participating_probes: {participating_probes}")
            print(f"[DEBUG] sorted_probe_ids: {sorted_probe_ids}")
            print(f"[DEBUG] peak_times: {peak_times}")
            # Ensure all sorted_probe_ids are in participating_probes
            if not all(pid in participating_probes for pid in sorted_probe_ids):
                print(f"[WARNING] Some sorted_probe_ids not in participating_probes. Skipping this event.")
                continue
            # Check if peak_times length matches participating_probes length
            if len(peak_times) != len(participating_probes):
                print(f"[WARNING] Mismatch between peak_times length ({len(peak_times)}) and participating_probes length ({len(participating_probes)}). Skipping this event.")
                continue
            sorted_peak_times = [peak_times[participating_probes.index(pid)] for pid in sorted_probe_ids]
            diffs = np.diff(sorted_peak_times)
            # Require all diffs to be at least min_peak_delay in the same direction
            if np.all(diffs >= min_peak_delay):
                directions.append('anterior')
            elif np.all(diffs <= -min_peak_delay):
                directions.append('posterior')
            else:
                directions.append('non_directional')
            # Power filter: all constituent probe events must be in range
            keep_mask.append(all((min_global_peak_power <= p < max_global_peak_power) for p in peak_powers))
            valid_events.append(idx)
        
        # Create a new DataFrame with only valid events
        filtered = global_events.loc[valid_events].copy()
        filtered['direction'] = directions
        filtered['all_probe_powers_in_range'] = keep_mask
        
        # --- Apply directionality filter FIRST ---
        if direction_filter == 'anterior':
            mask = filtered['direction'] == 'anterior'
        elif direction_filter == 'posterior':
            mask = filtered['direction'] == 'posterior'
        elif direction_filter == 'directional':
            mask = filtered['direction'].isin(['anterior', 'posterior'])
        elif direction_filter == 'non_directional':
            mask = filtered['direction'] == 'non_directional'
        else:
            mask = np.ones(len(filtered), dtype=bool)
        filtered = filtered[mask].copy()
        
        # --- Duration filter ---
        filtered = filtered[filtered['duration'] <= max_duration]
        # --- Power filter on constituent events ---
        filtered = filtered[filtered['all_probe_powers_in_range']]
        # Debug print: show peak_powers for each event after filtering
        print("Filtered global events (after power filter):")
        for idx, row in filtered.iterrows():
            print(f"Event {idx}: peak_powers={ast.literal_eval(row['peak_powers'])}")
        # --- Speed filter ---
        if 'running_speed' in filtered.columns:
            speeds = explorer.get_allensdk_speed(filtered['running_speed'], filtered, window=0.5, agg='max')
            filtered = filtered[speeds <= speed_threshold]
        
        # --- Now apply all other filters ---
        if len(filtered) > 0:
            filtered['dataset'] = dataset
            filtered['session_id'] = session_id
            # Get all probe IDs for the session, in AP order if possible
            probe_patterns = list(lfp_dir.glob("probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"))
            probe_ap_list = []
            probe_ap_dict = {}
            for lfp_file in probe_patterns:
                m_probe = lfp_file.name.split('_')[1]
                m_chan = lfp_file.name.split('_')[3]
                ap = explorer.get_channel_ap_coordinate(int(m_chan)) if m_chan.isdigit() else None
                probe_ap_list.append((m_probe, ap))
                probe_ap_dict[m_probe] = ap
            # Sort by AP coordinate (anterior-most first)
            probe_ap_list = sorted(probe_ap_list, key=lambda x: (x[1] if x[1] is not None else float('inf')))
            all_probes = [p[0] for p in probe_ap_list]
            # Compute contiguity score for each event
            contig_scores = []
            for idx, row in filtered.iterrows():
                participating = set(ast.literal_eval(row['participating_probes']))
                mask = [p in participating for p in all_probes]
                score = contiguity_score(mask)
                contig_scores.append(score)
            filtered['contiguity_score'] = contig_scores
            # Now apply probe count filter
            mask2 = (filtered['probe_count'] >= min_probe_count)
            filtered = filtered[mask2]
            all_candidates.append(filtered)
    if not all_candidates:
        print("No global events found matching criteria.")
        return pd.DataFrame()
    all_candidates_df = pd.concat(all_candidates)
    # Sort by contiguity score (descending), then by global_peak_power (descending)
    sorted_events = all_candidates_df.sort_values(['contiguity_score', 'global_peak_power'], ascending=[False, False])
    # Only keep the top N
    return sorted_events.head(target_count)

def plot_and_save_global_events(explorer, events_df, output_dir, file_ext='png'):
    os.makedirs(output_dir, exist_ok=True)
    for idx, (event_id, event) in enumerate(events_df.iterrows()):
        fig = explorer.plot_global_swr_event(
            dataset=event['dataset'],
            session_id=event['session_id'],
            global_event_idx=event_id,
            filter_path=filter_path,
            window=window,
            output_dir=output_dir,
            file_ext=file_ext,
            show_ap_in_ylabel=show_ap_in_ylabel,
            additional_value=additional_value,
            relative_time=relative_time
        )
        if fig is not None:
            plt.close(fig)
        # Only plot CSD if enabled
        if plot_csd_slices:
            csd_event_dir = os.path.join(
                output_dir,
                f"csd_slice_global_event_{idx+1}_session_{event['session_id']}_id_{event_id}"
            )
            os.makedirs(csd_event_dir, exist_ok=True)
            explorer.plot_global_event_CSD_slice(
                dataset=event['dataset'],
                session_id=event['session_id'],
                global_event_idx=event_id,
                filter_path=filter_path,
                window=window,
                save_path=csd_event_dir,  # Pass the event-specific CSD folder
                file_ext=file_ext
            )

def main():
    # Initialize the explorer with explicit base path
    base_path = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data"
    explorer = SWRExplorer(base_path=base_path)
    
    if collect_directional_stats:
        # Collect statistics about directional events
        session_stats = []
        for session_dir in (explorer.base_path / "allen_visbehave_swr_murphylab2024").glob("swrs_session_*"):
            session_id = session_dir.name.split('_')[-1]
            global_csvs = list(session_dir.glob(f"session_{session_id}_global_swr_events.csv.gz"))
            if not global_csvs:
                continue
            global_events = pd.read_csv(global_csvs[0], index_col=0)
            total_events = len(global_events)
            if total_events == 0:
                continue
            # Get AP coordinates for all probes in this session
            lfp_dir = explorer.base_path / explorer.lfp_sources["allen_visbehave_swr_murphylab2024"] / f"lfp_session_{session_id}"
            probe_ap_dict = {}
            for lfp_file in lfp_dir.glob("probe_*_channel*_lfp_ca1_putative_pyramidal_layer.npz"):
                m_probe = lfp_file.name.split('_')[1]
                m_chan = lfp_file.name.split('_')[3]
                ap = explorer.get_channel_ap_coordinate(int(m_chan)) if m_chan.isdigit() else None
                probe_ap_dict[m_probe] = ap
            # Count events by direction using new min_peak_delay logic
            anterior_count = 0
            posterior_count = 0
            non_directional_count = 0
            for _, event in global_events.iterrows():
                participating_probes = ast.literal_eval(event['participating_probes'])
                peak_times = ast.literal_eval(event['peak_times'])
                # Convert all probe IDs to strings for consistent comparison
                participating_probes = [str(pid) for pid in participating_probes]
                # Sort participating probes by AP coordinate
                participating_probes_ap = [(pid, probe_ap_dict.get(pid, float('inf'))) for pid in participating_probes]
                participating_probes_ap.sort(key=lambda x: x[1])
                sorted_probe_ids = [p[0] for p in participating_probes_ap]
                # Add error checking and debug info
                print(f"[DEBUG] participating_probes: {participating_probes}")
                print(f"[DEBUG] sorted_probe_ids: {sorted_probe_ids}")
                print(f"[DEBUG] peak_times: {peak_times}")
                # Ensure all sorted_probe_ids are in participating_probes
                if not all(pid in participating_probes for pid in sorted_probe_ids):
                    print(f"[WARNING] Some sorted_probe_ids not in participating_probes. Skipping this event.")
                    continue
                # Check if peak_times length matches participating_probes length
                if len(peak_times) != len(participating_probes):
                    print(f"[WARNING] Mismatch between peak_times length ({len(peak_times)}) and participating_probes length ({len(participating_probes)}). Skipping this event.")
                    continue
                sorted_peak_times = [peak_times[participating_probes.index(pid)] for pid in sorted_probe_ids]
                diffs = np.diff(sorted_peak_times)
                if np.all(diffs >= min_peak_delay):
                    anterior_count += 1
                elif np.all(diffs <= -min_peak_delay):
                    posterior_count += 1
                else:
                    non_directional_count += 1
            session_stats.append({
                'session_id': session_id,
                'total_events': total_events,
                'anterior_count': anterior_count,
                'posterior_count': posterior_count,
                'non_directional_count': non_directional_count,
                'anterior_percent': (anterior_count / total_events) * 100 if total_events > 0 else 0,
                'posterior_percent': (posterior_count / total_events) * 100 if total_events > 0 else 0,
                'non_directional_percent': (non_directional_count / total_events) * 100 if total_events > 0 else 0
            })
        # Print statistics
        print("\nDirectional Event Statistics:")
        print("-" * 100)
        print(f"{'Session ID':<15} {'Total':<10} {'Anterior':<15} {'Posterior':<15} {'Non-Dir':<15} {'A%':<10} {'P%':<10} {'ND%':<10}")
        print("-" * 100)
        total_events_all = 0
        total_anterior_all = 0
        total_posterior_all = 0
        total_non_directional_all = 0
        for stat in session_stats:
            print(f"{stat['session_id']:<15} {stat['total_events']:<10} {stat['anterior_count']:<15} {stat['posterior_count']:<15} {stat['non_directional_count']:<15} {stat['anterior_percent']:>6.1f}% {stat['posterior_percent']:>6.1f}% {stat['non_directional_percent']:>6.1f}%")
            total_events_all += stat['total_events']
            total_anterior_all += stat['anterior_count']
            total_posterior_all += stat['posterior_count']
            total_non_directional_all += stat['non_directional_count']
        if total_events_all > 0:
            print("-" * 100)
            print(f"{'Overall':<15} {total_events_all:<10} {total_anterior_all:<15} {total_posterior_all:<15} {total_non_directional_all:<15} {(total_anterior_all/total_events_all*100):>6.1f}% {(total_posterior_all/total_events_all*100):>6.1f}% {(total_non_directional_all/total_events_all*100):>6.1f}%")
    
    # Find and plot top global events
    top_global_events = find_top_global_events(
        explorer=explorer,
        dataset="allen_visbehave_swr_murphylab2024",
        min_probe_count=2,
        min_global_peak_power=5.0,
        max_global_peak_power=10.0,
        target_count=top_n_events,
        min_peak_delay=min_peak_delay
    )
    
    print(f"\nTop Global Events ({direction_filter}):")
    for idx, (event_id, event) in enumerate(top_global_events.iterrows()):
        print(f"Global Event {idx+1}:")
        print(f"  Session: {event['session_id']}")
        print(f"  Event ID: {event_id}")
        print(f"  Direction: {event['direction']}")
        print(f"  Probe Count: {event['probe_count']}")
        print(f"  Global Peak Power: {event['global_peak_power']:.2f}")
        print(f"  Participating Probes: {event['participating_probes']}")
        print(f"  Contiguity Score: {event['contiguity_score']}")
        print()
    
    plot_and_save_global_events(
        explorer=explorer,
        events_df=top_global_events,
        output_dir=output_dir,
        file_ext=file_ext
    )

if __name__ == "__main__":
    main() 