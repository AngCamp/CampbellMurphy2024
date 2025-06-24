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
OUTPUT_DIR = "yourpath/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Relating_SWR_to_other_data/Intuitive_Results"
SWR_INPUT_DIR = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
DATASET_NAME = "allen_visbehave_swr_murphylab2024"
TARGET_REGIONS = ['CA1', 'DG', 'SUB']
PER_REGION_UNIT_LIMITS_LIST = [30, 25, 10]  # Number of units per region, in the same order as TARGET_REGIONS
MIN_UNITS_PER_REGION = 100
BIN_SIZE = 0.01  # seconds
BASELINE_WINDOW = 0.25  # seconds around events to exclude from baseline
WINDOW = 0.5  # seconds around event to plot

# Debugging/limiting parameters
session_limit = 2      # Number of sessions to check (False/None for all)
unit_limit = False         # Number of units per region (False/None for all)
event_file_limit = 2   # Number of event files to load (False/None for all)
event_limit = 2        # Number of events to rank/plot (False/None for all)

# User-specified parameters for modular plotting
SESSION_ID = 1047969464  # Set to desired session
PROBE_ID = 1048089915    # Set to desired probe
EVENT_CSV_IDX = 2394     # Set to desired event index (row in CSV)
limit_to_probe = False    # Toggle: if True, only plot units from the specified probe; if False, plot all units in target regions



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
    for session_id in session_ids:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
        session = cache.get_ecephys_session(session_id)
        units = cache.get_unit_table()
        good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
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

def rank_units(session_id, cache_dir, swr_input_dir, target_regions, bin_size=BIN_SIZE, baseline_window=BASELINE_WINDOW, event_file_limit=None):
    """
    Analyze a single session, collect unit results, and return a dict and a flat list for FDR correction.
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
    time_bins = np.arange(session_start, session_end + bin_size, bin_size)
    n_bins = len(time_bins) - 1
    for region in target_regions:
        region_units = good_units[(good_units['ecephys_session_id'] == session_id) & (good_units['structure_acronym'] == region)]
        if region_units.empty:
            continue
        spike_matrix = np.zeros((len(region_units), n_bins), dtype=int)
        for idx, unit_id in enumerate(region_units.index):
            spikes = session.spike_times[unit_id]
            spike_matrix[idx], _ = np.histogram(spikes, bins=time_bins)
        during_mask = np.zeros(n_bins, dtype=bool)
        for _, event in session_events_df.iterrows():
            start_bin = int((event['start_time'] - session_start) / bin_size)
            end_bin = int((event['end_time'] - session_start) / bin_size)
            during_mask[start_bin:end_bin] = True
            baseline_start = max(0, start_bin - int(baseline_window / bin_size))
            baseline_end = min(n_bins, end_bin + int(baseline_window / bin_size))
            during_mask[baseline_start:baseline_end] = True
        during_samples = spike_matrix[:, during_mask]
        baseline_samples = spike_matrix[:, ~during_mask]
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

def plot_spike_raster_for_event(
    session, event, results_df, target_regions, window=WINDOW, align_to='peak', session_id=None,
    only_increasing=True, shaded_event_region=True, event_idx=None, probe_id=None, per_region_unit_limits=None
):
    # Initialize per_region_unit_limits if None
    if per_region_unit_limits is None:
        per_region_unit_limits = {region: None for region in target_regions}
    elif isinstance(per_region_unit_limits, list):
        # If a list is provided, convert to dict
        per_region_unit_limits = {region: per_region_unit_limits[i] for i, region in enumerate(target_regions)}
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
    plotted_any = False
    for region in reversed(target_regions):
        region_units = results_df[results_df['region'] == region]
        if only_increasing:
            region_units = region_units[region_units['direction'] == 'increase']
        # Sort by effect descending
        region_units = region_units.sort_values('effect', ascending=False)
        n_limit = per_region_unit_limits.get(region, None)
        if n_limit:
            region_units = region_units.head(n_limit)
        n_units = len(region_units)
        if n_units == 0:
            print(f"Warning: No significant units found for region {region} in this session.")
            region_boundaries.append(current_y)
            continue
        plotted_any = True
        region_units = region_units.reset_index(drop=True)
        for rank, unit_id in enumerate(region_units['unit_id']):
            flipped_rank = n_units - rank - 1
            spikes = session.spike_times[unit_id]
            rel_spikes = spikes[(spikes >= t_center - window/2) & (spikes <= t_center + window/2)] - t_center
            ax.vlines(rel_spikes, current_y + flipped_rank, current_y + flipped_rank + 0.8, color='red', linewidth=2)
        current_y += n_units
        region_boundaries.append(current_y)
    if not plotted_any:
        ax.text(0.5, 0.5, 'No units to plot', ha='center', va='center', transform=ax.transAxes)
        return fig
    for i, region in enumerate(reversed(target_regions)):
        if region_boundaries[i+1] > region_boundaries[i]:
            y_pos = (region_boundaries[i] + region_boundaries[i+1]) / 2
            ax.text(-window/2 - 0.02, y_pos, region, 
                    verticalalignment='center', horizontalalignment='right',
                    fontweight='bold')
        if i < len(target_regions) - 1:
            ax.axhline(region_boundaries[i+1] - 0.5, color='black', linestyle='-', linewidth=1)
    legend_handles = []
    legend_labels = []
    if shaded_event_region:
        span = ax.axvspan(event['start_time']-t_center, event['end_time']-t_center, color='green', alpha=0.2, label='Event Region')
        legend_handles.append(span)
        legend_labels.append('Event Region')
    if 'power_peak_time' in event:
        peak_line = ax.axvline(event['power_peak_time']-t_center, color='black', linestyle=':', linewidth=2)
        peak_marker = ax.plot([event['power_peak_time']-t_center], [current_y+1], marker='v', color='black', markersize=12, label='Event Peak')[0]
        legend_handles.append(peak_marker)
        legend_labels.append('Event Peak')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons', fontweight='bold')
    ax.set_yticks([])
    if event_idx is not None and probe_id is not None:
        ax.set_title(f'Spikes for units (session {session_id}, probe {probe_id}, event {event_idx})', fontweight='bold')
    elif event_idx is not None:
        ax.set_title(f'Spikes for units (session {session_id}, event {event_idx})', fontweight='bold')
    else:
        ax.set_title(f'Spikes for units (session {session_id}, event)', fontweight='bold')
    ax.set_ylim(-2, current_y + 2)
    plt.tight_layout()
    return fig

def rank_events_by_spike_count(session, swr_events, results_df, window=WINDOW):
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

def select_best_probe(results_df, units_table):
    """
    Vectorized: Given a DataFrame of significant units and the units table, find the probe with the most significant units.
    Returns the filtered results_df (only units on best probe), best_probe_id, and n_best.
    Uses 'ecephys_probe_id' from the cache units table.
    """
    # Merge results_df with units_table to get ecephys_probe_id for each unit
    merged = results_df.merge(
        units_table[['ecephys_probe_id']],
        left_on='unit_id',
        right_index=True,
        how='left'
    )
    merged = merged.rename(columns={'ecephys_probe_id': 'probe_id'})
    best_probe_id = merged['probe_id'].value_counts().idxmax()
    n_best = merged['probe_id'].value_counts().max()
    print(f"Selected probe {best_probe_id} with {n_best} significant increasing units.")
    filtered = merged[merged['probe_id'] == best_probe_id]
    return filtered, best_probe_id, n_best

def fdr_correct_and_summarize(flat_results):
    """
    Apply FDR correction to flat_results and return a summary DataFrame.
    """
    all_pvals = [x[4] for x in flat_results]
    _, qvals, _, _ = multipletests(all_pvals, method='fdr_bh')
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
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

def plot_specific_event(session_id, probe_id, event_csv_idx, cache_dir, swr_input_dir, target_regions, output_dir=OUTPUT_DIR, window=WINDOW, align_to='peak', limit_to_probe=True, per_region_unit_limits=None):
    """
    Plot a specific event for a given session, probe, and event index (row in CSV).
    If limit_to_probe is False, plot all good units in the target regions for the session (not just those on the probe).
    For each unit, compute effect as mean spike count in the event window for ranking.
    """
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units_table = cache.get_unit_table()
    session = cache.get_ecephys_session(session_id)
    # Filter units
    if limit_to_probe:
        good_units = units_table[
            (units_table['quality'] == 'good') &
            (units_table['valid_data'] == True) &
            (units_table['ecephys_session_id'] == session_id) &
            (units_table['ecephys_probe_id'] == probe_id) &
            (units_table['structure_acronym'].isin(target_regions))
        ]
    else:
        good_units = units_table[
            (units_table['quality'] == 'good') &
            (units_table['valid_data'] == True) &
            (units_table['ecephys_session_id'] == session_id) &
            (units_table['structure_acronym'].isin(target_regions))
        ]
    if good_units.empty:
        print(f"No good units found for session {session_id}, probe {probe_id}" if limit_to_probe else f"No good units found for session {session_id} in target regions.")
        return
    # Load the event file(s) for the specified probe
    session_path = os.path.join(swr_input_dir, DATASET_NAME, f"swrs_session_{session_id}")
    event_file_pattern = os.path.join(session_path, f"probe_{probe_id}_channel_*_putative_swr_events.csv.gz")
    event_files = glob.glob(event_file_pattern)
    if not event_files:
        print(f"No event files found for probe {probe_id} in session {session_id}")
        return
    # Concatenate all event files for this probe (usually one, but just in case)
    all_events = []
    for event_file in event_files:
        events = pd.read_csv(event_file, compression='gzip')
        if not events.empty:
            all_events.append(events)
    if not all_events:
        print(f"No events found for probe {probe_id} in session {session_id}")
        return
    swr_events = pd.concat(all_events, ignore_index=True)
    if event_csv_idx not in swr_events.index:
        print(f"Event index {event_csv_idx} not found in event file for probe {probe_id}")
        return
    event = swr_events.loc[event_csv_idx]
    # Compute effect for each unit: mean spike count in event window
    t_center = (event['start_time'] + event['end_time']) / 2
    event_window_start = t_center - window/2
    event_window_end = t_center + window/2
    effect_list = []
    for unit_id, row in good_units.iterrows():
        spikes = session.spike_times[unit_id]
        n_spikes = np.sum((spikes >= event_window_start) & (spikes <= event_window_end))
        effect_list.append(n_spikes)
    results_df = pd.DataFrame({
        'unit_id': good_units.index,
        'region': good_units['structure_acronym'],
        'effect': effect_list,
        'direction': ['increase'] * len(good_units)  # For compatibility
    })
    fig = plot_spike_raster_for_event(
        session, event, results_df, target_regions, window=window, align_to=align_to,
        session_id=session_id, only_increasing=True, shaded_event_region=True, event_idx=event_csv_idx, probe_id=probe_id,
        per_region_unit_limits=per_region_unit_limits
    )
    base_path = os.path.join(output_dir, f'swr_spike_raster_session_{session_id}_probe_{probe_id}_event_{event_csv_idx}')
    fig.savefig(base_path + '.png')
    fig.savefig(base_path + '.svg')
    plt.close(fig)
    print(f"Plot saved for event {event_csv_idx} to {base_path}.png and .svg")

def main():
    # User provides session_id
    session_id = 1047969464  # Example, replace as needed
    results_dict, flat_results = rank_units(session_id, CACHE_DIR, SWR_INPUT_DIR, TARGET_REGIONS)
    if not flat_results:
        print("No valid unit results found for this session.")
        return
    # FDR correction and summary
    summary_df = fdr_correct_and_summarize(flat_results)
    # Find best probe by number of significant increasing units
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
    units_table = cache.get_unit_table()
    results_df = summary_df[summary_df['direction'] == 'increase']
    if results_df.empty:
        print("No significant increasing units found for this session.")
        return
    # Use helper to select best probe
    results_df, best_probe_id, n_best = select_best_probe(results_df, units_table)
    session = cache.get_ecephys_session(session_id)
    session_path = os.path.join(SWR_INPUT_DIR, DATASET_NAME, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    all_events = []
    for event_file in event_files:
        events = pd.read_csv(event_file, compression='gzip')
        if not events.empty:
            all_events.append(events)
    if not all_events:
        print(f"No SWR events found for session {session_id}")
        return
    swr_events = pd.concat(all_events, ignore_index=True)
    # Rank events by spike count (using only units from best probe)
    ranked_df = rank_events_by_spike_count(session, swr_events, results_df, window=window)
    # Plot the best event
    if ranked_df.empty:
        print("No events to plot.")
        return
    best_event = ranked_df.iloc[0]
    fig = plot_spike_raster_for_event(
        session, best_event, results_df, TARGET_REGIONS, window=window, align_to='peak', session_id=session_id,
        only_increasing=True, shaded_event_region=True, event_idx=best_event.event_csv_idx, probe_id=best_probe_id,
        per_region_unit_limits=PER_REGION_UNIT_LIMITS_LIST
    )
    base_path = os.path.join(OUTPUT_DIR, f'swr_spike_raster_session_{session_id}_probe_{best_probe_id}_event_{best_event.event_csv_idx}')
    fig.savefig(base_path + '.png')
    fig.savefig(base_path + '.svg')
    plt.close(fig)
    print(f"Plot saved for event {best_event.event_csv_idx} to {base_path}.png and .svg")

if __name__ == "__main__":
    # Uncomment the following to use the modular event plotting:
    plot_specific_event(SESSION_ID, PROBE_ID, EVENT_CSV_IDX, CACHE_DIR, SWR_INPUT_DIR, TARGET_REGIONS, window=WINDOW, limit_to_probe=limit_to_probe, per_region_unit_limits=PER_REGION_UNIT_LIMITS_LIST)
    # Or comment above and use the default main workflow:
    # main()