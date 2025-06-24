#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
import os
import sys
import glob
import re
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import SWRExplorer
try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from Sharp_wave_component_validation.SWRExplorer import SWRExplorer

# Configuration Parameters
CACHE_DIR = "yourpath/allen_visbehave_data"
OUTPUT_DIR = "yourpath/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Relating_SWR_to_other_data/Results"
SWR_INPUT_DIR = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
DATASET_NAME = "allen_visbehave_swr_murphylab2024"
TARGET_REGIONS = ['CA1', 'DG', 'SUB']
PER_REGION_UNIT_LIMITS_LIST = [30, 25, 10]  # Number of units per region, in the same order as TARGET_REGIONS
MIN_UNITS_PER_REGION = 100
BIN_SIZE = 0.01  # seconds
BASELINE_WINDOW = 0.2  # seconds around events to exclude from baseline

# Debugging/limiting parameters
session_limit = 10      # Number of sessions to check (False/None for all)
unit_limit = False         # Number of units per region (False/None for all)
event_file_limit = None   # Number of event files to load (False/None for all)
event_limit = 10        # Number of events to rank/plot (False/None for all)

# Add this flag at the top of the script
USE_MULTIUNIT_SESSION_RANKING = True  # Set to True to use fast multi-unit session ranking

def find_sessions_with_units(cache_dir, target_regions, min_units_per_region, session_limit=session_limit):
    """Find sessions with enough good units in target regions. Limit to session_limit if set."""
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units = cache.get_unit_table()
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
    roi_units = good_units[good_units['structure_acronym'].isin(target_regions)]
    grouped_counts = (
        roi_units
        .groupby(['ecephys_session_id', 'structure_acronym'])
        .size()
        .unstack(fill_value=0)
    )
    valid_sessions = grouped_counts[
        grouped_counts[target_regions].ge(min_units_per_region).all(axis=1)
    ]
    if session_limit:
        valid_sessions = valid_sessions.head(session_limit)
    return valid_sessions

def calculate_unit_significance_from_events(session, unit_id, events_df, bin_size=BIN_SIZE):
    """Calculate Mann-Whitney U test for a unit's firing during vs outside SWR events loaded from CSV, with event quality filtering."""
    spike_times = session.spike_times[unit_id]
    if events_df.empty:
        return np.nan, np.nan
        
    # Get session path to find gamma band event files
    session_id = events_df['session_id'].iloc[0] if 'session_id' in events_df.columns else None
    if session_id is None:
        print(f"Warning: No session_id found in events_df")
        return np.nan, np.nan
        
    session_path = os.path.join(SWR_INPUT_DIR, DATASET_NAME, f"swrs_session_{session_id}")
    gamma_event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_gamma_band_events.csv.gz"))
    
    # Load all gamma band events
    gamma_events = []
    for gamma_file in gamma_event_files:
        try:
            gamma_df = pd.read_csv(gamma_file, compression='gzip')
            if not gamma_df.empty:
                gamma_events.append(gamma_df)
        except Exception as e:
            print(f"Warning: Could not load gamma file {gamma_file}: {e}")
            continue
            
    # Filter events for 'during' mask - only high quality SWR events
    good_events = events_df[
        (events_df['power_max_zscore'] >= 3) & (events_df['power_max_zscore'] <= 10)
        & (events_df['sw_peak_power'] > 1)
        & (~events_df['overlaps_with_gamma'])
        & (~events_df['overlaps_with_movement'])
    ]
    
    # For baseline exclusion, exclude:
    # 1. All SWR events (any power_max_zscore >= 0)
    # 2. Events with gamma/movement overlap
    # 3. Additional gamma band events
    exclude_events = events_df[events_df['power_max_zscore'] >= 0]
    
    # Add gamma band events to exclusion
    if gamma_events:
        gamma_events_df = pd.concat(gamma_events, ignore_index=True)
        exclude_events = pd.concat([exclude_events, gamma_events_df], ignore_index=True)
    
    session_start = events_df['start_time'].min()
    session_end = events_df['end_time'].max()
    time_bins = np.arange(session_start, session_end + bin_size, bin_size)
    spike_counts = np.histogram(spike_times, bins=time_bins)[0]
        
    # Create during mask
    during_mask = np.zeros_like(spike_counts, dtype=bool)
    for _, event in good_events.iterrows():
        start_bin = int((event['start_time'] - session_start) / bin_size)
        end_bin = int((event['end_time'] - session_start) / bin_size)
        during_mask[start_bin:end_bin] = True
    
    # Create exclude mask for baseline
    exclude_mask = np.zeros_like(spike_counts, dtype=bool)
    for _, event in exclude_events.iterrows():
        start_bin = int((event['start_time'] - session_start) / bin_size)
        end_bin = int((event['end_time'] - session_start) / bin_size)
        exclude_mask[start_bin:end_bin] = True
    
    baseline_mask = ~during_mask & ~exclude_mask
    during_samples = spike_counts[during_mask]
    baseline_samples = spike_counts[baseline_mask]
    
    try:
        stat, pval = mannwhitneyu(during_samples, baseline_samples, alternative='two-sided')
        effect = np.mean(during_samples) - np.mean(baseline_samples)
        return effect, pval
    except:
        return np.nan, np.nan

def rank_units(session_id, cache_dir, swr_input_dir, target_regions, unit_limit=unit_limit, event_file_limit=event_file_limit):
    """
    Analyze a single session, collect unit results, and return a nested dict and a flat list for FDR correction.
    Loads SWR events from probe event CSVs for the session.
        Returns:
        results_dict: {unit_id: {'region': ..., 'effect': ..., 'pval': ...}}
        flat_results: list of (session_id, unit_id, region, effect, pval)
    """
    results_dict = dict()
    flat_results = []
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units = cache.get_unit_table()
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
    session = cache.get_ecephys_session(session_id)
    session_path = os.path.join(swr_input_dir, DATASET_NAME, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    all_events = []
    for i, event_file in enumerate(event_files):
        if event_file_limit and i >= event_file_limit:
            break
        try:
            events_df = pd.read_csv(event_file, compression='gzip')
            if not events_df.empty:
                all_events.append(events_df)
        except Exception as e:
            print(f"Warning: Could not load {event_file}: {e}")
            continue
    if not all_events:
        return results_dict, flat_results
    session_events_df = pd.concat(all_events, ignore_index=True)
    session_start = session_events_df['start_time'].min()
    session_end = session_events_df['end_time'].max()
    time_bins = np.arange(session_start, session_end + BIN_SIZE, BIN_SIZE)
    n_bins = len(time_bins) - 1

    for region in target_regions:
        region_units = good_units[(good_units['ecephys_session_id'] == session_id) & (good_units['structure_acronym'] == region)]
        if unit_limit:
            region_units = region_units.head(unit_limit)
        if region_units.empty:
            continue
        # Vectorized spike binning
        spike_matrix = np.zeros((len(region_units), n_bins), dtype=int)
        for idx, unit_id in enumerate(region_units.index):
            spikes = session.spike_times[unit_id]
            spike_matrix[idx], _ = np.histogram(spikes, bins=time_bins)
        # Build event mask
        during_mask = np.zeros(n_bins, dtype=bool)
        for _, event in session_events_df.iterrows():
            start_bin = int((event['start_time'] - session_start) / BIN_SIZE)
            end_bin = int((event['end_time'] - session_start) / BIN_SIZE)
            during_mask[start_bin:end_bin] = True
            baseline_start = max(0, start_bin - int(BASELINE_WINDOW / BIN_SIZE))
            baseline_end = min(n_bins, end_bin + int(BASELINE_WINDOW / BIN_SIZE))
            during_mask[baseline_start:baseline_end] = True
        # Vectorized mean calculation
        during_samples = spike_matrix[:, during_mask]
        baseline_samples = spike_matrix[:, ~during_mask]
        # Parallel Mann-Whitney U test
        def unit_test(idx_unit):
            try:
                stat, pval = mannwhitneyu(during_samples[idx_unit], baseline_samples[idx_unit], alternative='two-sided')
                effect = np.mean(during_samples[idx_unit]) - np.mean(baseline_samples[idx_unit])
            except Exception:
                effect, pval = np.nan, np.nan
            return (region_units.index[idx_unit], effect, pval)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(unit_test, range(len(region_units))))
        for unit_id, effect, pval in results:
            if not np.isnan(effect):
                results_dict[unit_id] = {'region': region, 'effect': effect, 'pval': pval}
                flat_results.append((session_id, unit_id, region, effect, pval))
    return results_dict, flat_results

def analyze_all_sessions(session_ids, cache_dir, swr_input_dir, target_regions, unit_limit=unit_limit, event_file_limit=event_file_limit):
    """
    Analyze all sessions, collect unit results, and return a nested dict and a flat list for FDR correction.
    Loads SWR events from probe event CSVs for each session.
        Returns:
        results_dict: {session_id: {unit_id: {'region': ..., 'effect': ..., 'pval': ...}}}
        flat_results: list of (session_id, unit_id, region, effect, pval)
    """
    results_dict = defaultdict(dict)
    flat_results = []
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units = cache.get_unit_table()
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
    for session_id in session_ids:
        session = cache.get_ecephys_session(session_id)
        session_path = os.path.join(swr_input_dir, DATASET_NAME, f"swrs_session_{session_id}")
        event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
        all_events = []
        for i, event_file in enumerate(event_files):
            if event_file_limit and i >= event_file_limit:
                break
            try:
                events_df = pd.read_csv(event_file, compression='gzip')
                if not events_df.empty:
                    all_events.append(events_df)
            except Exception as e:
                print(f"Warning: Could not load {event_file}: {e}")
                continue
        if not all_events:
            continue
        # Concatenate all probe events for this session
        session_events_df = pd.concat(all_events, ignore_index=True)
        for region in target_regions:
            region_units = good_units[(good_units['ecephys_session_id'] == session_id) & (good_units['structure_acronym'] == region)]
            for i, unit_id in enumerate(region_units.index):
                if unit_limit and i >= unit_limit:
                    break
                effect, pval = calculate_unit_significance_from_events(session, unit_id, session_events_df)
                if not np.isnan(effect):
                    results_dict[session_id][unit_id] = {'region': region, 'effect': effect, 'pval': pval}
                    flat_results.append((session_id, unit_id, region, effect, pval))
    return results_dict, flat_results

def plot_spike_raster_for_event(
    session, event, results_df, target_regions, window=0.5, align_to='peak', session_id=None,
    per_region_unit_limits_list=None, only_increasing=True, shaded_event_region=True, event_idx=None
):
    """
    Plot a raster of spikes for a given event, with regions and units ordered as specified.
    Parameters:
        session: AllenSDK session object
        event: pd.Series (row from swr_events DataFrame)
        results_df: DataFrame with columns ['unit_id', 'region', 'effect', ...]
        target_regions: list of str, order of regions (top to bottom)
        window: float, window size in seconds (total width)
        align_to: 'start', 'center', or 'peak' (where to center the window)
        session_id: int, session ID for the title
        per_region_unit_limits_list: list of int or None, neuron limits in the same order as target_regions
        only_increasing: bool, if True only plot significant increasing neurons
        shaded_event_region: bool, if True shade event region instead of arrows
        event_idx: int or None, event index for the title
    """
    # Build per-region unit limits dict if provided
    per_region_unit_limits = {}
    if per_region_unit_limits_list is not None:
        for i, region in enumerate(target_regions):
            if i < len(per_region_unit_limits_list):
                per_region_unit_limits[region] = per_region_unit_limits_list[i]
            else:
                per_region_unit_limits[region] = None
    else:
        for region in target_regions:
            per_region_unit_limits[region] = None

    # Determine event alignment time
    if align_to == 'start':
        t_center = event['start_time']
    elif align_to == 'end':
        t_center = event['end_time']
    elif align_to == 'peak' and 'power_peak_time' in event:
        t_center = event['power_peak_time']
    else:
        t_center = (event['start_time'] + event['end_time']) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.margins(x=0, y=0)
    current_y = 0
    region_boundaries = [0]
    region_yticks = []
    region_yticklabels = []
    plotted_any = False

    # Reverse the region order for plotting so the first region is at the top
    for region in reversed(target_regions):
        # Filter units for this region
        region_units = results_df[results_df['region'] == region]
        if only_increasing:
            region_units = region_units[region_units['direction'] == 'increase']
        # Sort by effect (descending)
        region_units = region_units.sort_values('effect', ascending=False)
        # Apply per-region unit limit if provided
        n_limit = per_region_unit_limits.get(region, None)
        if n_limit:
            region_units = region_units.head(n_limit)
        n_units = len(region_units)
        if n_units == 0:
            print(f"Warning: No significant units found for region {region} in this session.")
            region_boundaries.append(current_y)
            continue
        plotted_any = True
        # Flip the order so top neuron is at the top within the region
        region_units = region_units.reset_index(drop=True)
        for rank, unit_id in enumerate(region_units['unit_id']):
            flipped_rank = n_units - rank - 1
            spikes = session.spike_times[unit_id]
            rel_spikes = spikes[(spikes >= t_center - window/2) & (spikes <= t_center + window/2)] - t_center
            ax.vlines(rel_spikes, current_y + flipped_rank, current_y + flipped_rank + 0.8, color='red', linewidth=2)
        # For yticks/labels
        region_yticks.append(current_y + n_units / 2)
        region_yticklabels.append(region)
        current_y += n_units
        region_boundaries.append(current_y)

    if not plotted_any:
        ax.text(0.5, 0.5, 'No units to plot', ha='center', va='center', transform=ax.transAxes)
        return fig

    # Add region boundaries and labels (reverse for correct y-label order)
    for i, region in enumerate(reversed(target_regions)):
        if region_boundaries[i+1] > region_boundaries[i]:
            y_pos = (region_boundaries[i] + region_boundaries[i+1]) / 2
            ax.text(-window/2 - 0.02, y_pos, region, 
                    verticalalignment='center', horizontalalignment='right',
                    fontweight='bold')
        if i < len(target_regions) - 1:
            ax.axhline(region_boundaries[i+1] - 0.5, color='black', linestyle='-', linewidth=1)

    # Mark event region and peak
    legend_handles = []
    legend_labels = []
    if shaded_event_region:
        span = ax.axvspan(event['start_time']-t_center, event['end_time']-t_center, color='green', alpha=0.2, label='Event Region')
        legend_handles.append(span)
        legend_labels.append('Event Region')
    else:
        start_marker = ax.plot([event['start_time']-t_center], [-1], marker='v', color='green', markersize=12, label='Event Start')[0]
        end_marker = ax.plot([event['end_time']-t_center], [-1], marker='v', color='green', markersize=12, label='Event End')[0]
        legend_handles.extend([start_marker, end_marker])
        legend_labels.extend(['Event Start', 'Event End'])
    # Black dotted line for peak
    if 'power_peak_time' in event:
        peak_line = ax.axvline(event['power_peak_time']-t_center, color='black', linestyle=':', linewidth=2)
        # Peak triangle on top
        peak_marker = ax.plot([event['power_peak_time']-t_center], [current_y+1], marker='v', color='black', markersize=12, label='Event Peak')[0]
        legend_handles.append(peak_marker)
        legend_labels.append('Event Peak')
    # Center line (dashed)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right')

    ax.set_xlabel('Time (s)')
    # Set y-axis label only (no ticks)
    ax.set_ylabel('Neurons', fontweight='bold')
    ax.set_yticks([])
    # Bold title, include event index
    if event_idx is not None:
        ax.set_title(f'Spikes for units (session {session_id}, event {event_idx})', fontweight='bold')
    else:
        ax.set_title(f'Spikes for units (session {session_id}, event)', fontweight='bold')
    ax.set_ylim(-2, current_y + 2)
    plt.tight_layout()
    return fig

def find_and_rank_events(session, swr_events, results_df, window=0.5, event_limit=event_limit):
    """
    For each event, count total spikes from significant units within the window.
    Returns a DataFrame of events ranked by total spike count (descending). Limit to event_limit events if set.
    """
    ranked_events = []
    for idx, event in swr_events.iterrows():
        if event_limit and idx >= event_limit:
            break
        total_spikes = 0
        for unit_id in results_df['unit_id']:
            spikes = session.spike_times[unit_id]
            mask = (spikes >= event['start_time'] - window/2) & (spikes <= event['end_time'] + window/2)
            total_spikes += np.sum(mask)
        ranked_events.append({**event, 'event_idx': idx, 'total_spikes': total_spikes})
    ranked_df = pd.DataFrame(ranked_events)
    ranked_df = ranked_df.sort_values('total_spikes', ascending=False).reset_index(drop=True)
    return ranked_df

def get_session_ids_from_swr_folder(swr_input_dir, dataset_name, session_limit=None):
    """Glob the SWR data directory for session folders and extract session IDs."""
    session_dir = os.path.join(swr_input_dir, dataset_name)
    session_folders = glob.glob(os.path.join(session_dir, 'swrs_session_*'))
    session_ids = []
    for folder in session_folders:
        match = re.search(r'swrs_session_(\d+)', os.path.basename(folder))
        if match:
            session_ids.append(int(match.group(1)))
    session_ids = sorted(session_ids)
    if session_limit:
        session_ids = session_ids[:session_limit]
    return session_ids

# Add this function for multi-unit session ranking
def rank_sessions_by_multiunit(cache_dir, swr_input_dir, dataset_name, target_regions, session_ids, bin_size=BIN_SIZE, baseline_window=BASELINE_WINDOW):
    """
    For each session, sum all spikes from all good units in the selected regions into time bins,
    then perform the event vs. baseline Mann-Whitney U test on these multi-unit binned spike counts.
    Returns a DataFrame with session_id, effect, pval, and qval (FDR-corrected p-value).
    """
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units = cache.get_unit_table()
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
    roi_units = good_units[good_units['structure_acronym'].isin(target_regions)]
    results = []
    for session_id in session_ids:
        session_units = roi_units[roi_units['ecephys_session_id'] == session_id]
        if session_units.empty:
            continue
        session = cache.get_ecephys_session(session_id)
        session_path = os.path.join(swr_input_dir, dataset_name, f"swrs_session_{session_id}")
        event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
        all_events = []
        for event_file in event_files:
            try:
                events_df = pd.read_csv(event_file, compression='gzip')
                if not events_df.empty:
                    all_events.append(events_df)
            except Exception as e:
                continue
        if not all_events:
            continue
        events_df = pd.concat(all_events, ignore_index=True)
        # Get all spike times for all units in this session/regions
        all_spike_times = np.concatenate([
            session.spike_times[unit_id] for unit_id in session_units.index
        ])
        if len(all_spike_times) == 0:
            continue
        # Determine session duration from event times
        session_start = events_df['start_time'].min()
        session_end = events_df['end_time'].max()
        time_bins = np.arange(session_start, session_end + bin_size, bin_size)
        spike_counts = np.histogram(all_spike_times, bins=time_bins)[0]
        # Create masks for during-event and baseline periods
        during_mask = np.zeros_like(spike_counts, dtype=bool)
        for _, event in events_df.iterrows():
            start_bin = int((event['start_time'] - session_start) / bin_size)
            end_bin = int((event['end_time'] - session_start) / bin_size)
            during_mask[start_bin:end_bin] = True
            baseline_start = max(0, start_bin - int(baseline_window / bin_size))
            baseline_end = min(len(during_mask), end_bin + int(baseline_window / bin_size))
            during_mask[baseline_start:baseline_end] = True
        during_samples = spike_counts[during_mask]
        baseline_samples = spike_counts[~during_mask]
        try:
            stat, pval = mannwhitneyu(during_samples, baseline_samples, alternative='two-sided')
            effect = np.mean(during_samples) - np.mean(baseline_samples)
        except:
            effect, pval = np.nan, np.nan
        results.append({'session_id': session_id, 'effect': effect, 'pval': pval})
    # FDR correction
    df = pd.DataFrame(results)
    if not df.empty:
        _, qvals, _, _ = multipletests(df['pval'].fillna(1.0), method='fdr_bh')
        df['qval'] = qvals
    else:
        df['qval'] = []
    return df

def rank_events_by_spike_count(session, swr_events, results_df, window=0.5, event_limit=None):
    """
    For each event, count total spikes from significant units within the window.
    Returns a DataFrame of events ranked by total spike count (descending), with the original event CSV index as 'event_csv_idx'.
    """
    ranked_events = []
    for idx, event in swr_events.iterrows():
        total_spikes = 0
        for unit_id in results_df['unit_id']:
            spikes = session.spike_times[unit_id]
            mask = (spikes >= event['start_time'] - window/2) & (spikes <= event['end_time'] + window/2)
            total_spikes += np.sum(mask)
        ranked_events.append({**event, 'event_csv_idx': event.name, 'total_spikes': total_spikes})
    ranked_df = pd.DataFrame(ranked_events)
    ranked_df = ranked_df.sort_values('total_spikes', ascending=False).reset_index(drop=True)
    if event_limit:
        ranked_df = ranked_df.head(event_limit)
    return ranked_df

def main():
    """Main function to find best session and create plots for top 10 events."""
    # Get session IDs from SWR data folder
    session_ids = get_session_ids_from_swr_folder(SWR_INPUT_DIR, DATASET_NAME, session_limit=session_limit)
    if not session_ids:
        print("No SWR session folders found in the data directory.")
        return
    print(f"Analyzing sessions (from SWR data): {session_ids}")
    if USE_MULTIUNIT_SESSION_RANKING:
        ranked_df = rank_sessions_by_multiunit(CACHE_DIR, SWR_INPUT_DIR, DATASET_NAME, TARGET_REGIONS, session_ids)
        if ranked_df.empty:
            print("No valid multi-unit session results found.")
            return
        # Pick the best session (highest effect, qval < 0.05)
        ranked_df = ranked_df.sort_values(['qval', 'effect'], ascending=[True, False])
        best_sessions = ranked_df[ranked_df['qval'] < 0.05]
        if best_sessions.empty:
            print("No sessions passed FDR threshold, using top effect session.")
            session_id = int(ranked_df.iloc[0]['session_id'])
        else:
            session_id = int(best_sessions.iloc[0]['session_id'])
        print(f"Selected session {session_id} for detailed analysis.")
        session_ids = [session_id]
        results_dict, flat_results = rank_units(session_id, CACHE_DIR, SWR_INPUT_DIR, TARGET_REGIONS, unit_limit=unit_limit, event_file_limit=event_file_limit)
    else:
        results_dict = dict()
        flat_results = []
        for session_id in session_ids:
            session_results, session_flat = rank_units(session_id, CACHE_DIR, SWR_INPUT_DIR, TARGET_REGIONS, unit_limit=unit_limit, event_file_limit=event_file_limit)
            # Store results by session
            results_dict[session_id] = session_results
            flat_results.extend(session_flat)
    if not flat_results:
        print("No valid unit results found for any session.")
        return
    # Apply FDR correction globally
    all_pvals = [x[4] for x in flat_results]
    _, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')
    # Build summary DataFrame
    summary_rows = []
    for (session_id, unit_id, region, effect, pval), qval in zip(flat_results, qvals):
        if qval < 0.05 and effect > 0:
            direction = 'increase'
        elif qval < 0.05 and effect < 0:
            direction = 'decrease'
        else:
            direction = 'none'
        summary_rows.append({
            'session_id': session_id,
            'unit_id': unit_id,
            'region': region,
            'effect': effect,
            'pval': pval,
            'qval': qval,
            'direction': direction
        })
        # Store qval and direction in results_dict for later use
        if USE_MULTIUNIT_SESSION_RANKING:
            results_dict[unit_id]['qval'] = qval
            results_dict[unit_id]['direction'] = direction
        else:
            results_dict[session_id][unit_id]['qval'] = qval
            results_dict[session_id][unit_id]['direction'] = direction
        summary_df = pd.DataFrame(summary_rows)
    # Save summary DataFrame as CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_csv = os.path.join(OUTPUT_DIR, 'unit_significance_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved unit significance summary to {summary_csv}")
    # For plotting, proceed as before for the first session
    session_id = session_ids[0]
    results_df = summary_df[summary_df['session_id'] == session_id]
    # Get session and SWR events
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
    session = cache.get_ecephys_session(session_id)
    explorer = SWRExplorer(base_path=SWR_INPUT_DIR)
    session_path = os.path.join(SWR_INPUT_DIR, DATASET_NAME, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    all_events = []
    for i, event_file in enumerate(event_files):
        if event_file_limit and i >= event_file_limit:
            break
        match = re.search(r'probe_([^_]+)_channel_', os.path.basename(event_file))
        if match:
            probe_id = match.group(1)
            events = explorer.find_best_events(
                dataset=DATASET_NAME,
                session_id=str(session_id),
                probe_id=str(probe_id),
                min_sw_power=1,
                min_duration=0.05,
                max_duration=0.15,
                min_clcorr=0,
                exclude_gamma=True,
                exclude_movement=True
            )
            if not events.empty:
                all_events.append(events)
    if not all_events:
        print(f"No SWR events found for session {session_id}")
        return
    swr_events = pd.concat(all_events, ignore_index=True)
    # Rank events by spike count (using new function)
    ranked_events = rank_events_by_spike_count(session, swr_events, results_df, window=0.5, event_limit=event_limit)
    # Plot top n events
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_plot = event_limit if event_limit else 10
    for i, event in ranked_events.head(n_plot).iterrows():
        fig = plot_spike_raster_for_event(
            session, event, results_df, TARGET_REGIONS, window=0.5, align_to='peak', session_id=session_id,
            per_region_unit_limits_list=PER_REGION_UNIT_LIMITS_LIST, only_increasing=True, shaded_event_region=True, event_idx=event.event_csv_idx
        )
        base_path = os.path.join(OUTPUT_DIR, f'swr_spike_raster_session_{session_id}_event_{event.event_csv_idx}')
        fig.savefig(base_path + '.png')
        fig.savefig(base_path + '.svg')
        plt.close(fig)
        print(f"Plot saved for event {event.event_csv_idx} to {base_path}.png and .svg")

if __name__ == "__main__":
    main()