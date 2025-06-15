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

# Add parent directory to path to import SWRExplorer
try:
    # This works in script context
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # This works in notebook context
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from Sharp_wave_component_validation.SWRExplorer import SWRExplorer

from scipy import signal
from scipy.stats import zscore
import seaborn as sns
from scipy.signal import hilbert
import argparse
import json
from datetime import datetime
import logging

# =============================================================================
# Configuration Parameters
# =============================================================================
# Cache and data paths
CACHE_DIR = "/space/scratch/allen_visbehave_data"
OUTPUT_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_and_Technical_Validation/Relating_SWR_to_other_data/Results"
SWR_INPUT_DIR = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"  # Directory containing SWR event files

# Dataset configuration
DATASET_NAME = "allen_visbehave_swr_murphylab2024"  # Name of the dataset in SWRExplorer

# Session finding parameters
MIN_UNITS_PER_REGION = 100  # Minimum number of units required in each region
MAX_SPEED_THRESHOLD = 5.0  # Maximum speed during SWR (cm/s)
MIN_PUPIL_DIAMETER = 0.5  # Minimum pupil diameter (arbitrary units)
MAX_PUPIL_DIAMETER = 2.0  # Maximum pupil diameter (arbitrary units)
EVENTS_PER_SESSION = 3  # Number of best events to find per session

# SWR detection parameters
MIN_SW_POWER = 1
MIN_DURATION = 0.05
MAX_DURATION = 0.15
WINDOW_SIZE = 0.2  # Window size for spike correlation (seconds)

# Ripple band power parameters
MIN_RIPPLE_POWER = 5.0  # Minimum ripple band peak power (z-score)
MAX_RIPPLE_POWER = 10.0  # Maximum ripple band peak power (z-score)

# Target regions to analyze
TARGET_REGIONS = ['RSC', 'SUB']

def abi_visual_behavior_units_session_search(
    cache_dir,
    target_regions,
    min_units_per_region=5
):
    """
    From the Allen Visual Behavior Neuropixels dataset, find sessions with at least
    `min_units_per_region` good units in any of the `target_regions`.

    Parameters
    ----------
    cache_dir : str
        Path to the AllenSDK VisualBehaviorNeuropixelsProjectCache directory.
    target_regions : list of str
        List of structure acronyms to consider as regions of interest.
    min_units_per_region : int, default=5
        Minimum number of good units required in any one region.

    Returns
    -------
    passed_sessions_df : pd.DataFrame
        Session-by-region table with good unit counts, indexed by session ID,
        plus a column with total units across all target regions.
    """
    # Load cache and tables
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
    units = cache.get_unit_table()

    print("\nInitial data shapes:")
    print(f"Units table shape: {units.shape}")

    # Filter for good units
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]

    # Filter for target regions
    roi_units = good_units[good_units['structure_acronym'].isin(target_regions)]

    # Group by session and region, count units
    grouped_counts = (
        roi_units
        .groupby(['ecephys_session_id', 'structure_acronym'])
        .size()
        .unstack(fill_value=0)
    )

    # Add total across all target regions
    grouped_counts['total_good_units_in_rois'] = grouped_counts.sum(axis=1)

    # Sort by RSC count (descending)
    if 'RSC' in grouped_counts.columns:
        grouped_counts = grouped_counts.sort_values('RSC', ascending=False)

    print(f"\nNumber of sessions with units in target regions: {len(grouped_counts)}")
    return grouped_counts

class SWRSpikeAnalyzer:
    def __init__(self, cache_dir, swr_input_dir, dataset_name="allen_visbehave_swr_murphylab2024"):
        """
        Initialize the SWRSpikeAnalyzer.
        
        Parameters:
        -----------
        cache_dir : str
            Path to the AllenSDK cache directory
        swr_input_dir : str
            Path to the directory containing SWR event files
        dataset_name : str
            Name of the dataset in SWRExplorer
        """
        self.cache_dir = cache_dir
        self.swr_input_dir = swr_input_dir
        self.dataset_name = dataset_name
        self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)
        self.explorer = SWRExplorer(base_path=swr_input_dir)
        
    def get_region_units_and_spikes(self, session_id, region):
        """
        Get units and spike times for a specific region in a session.
        
        Parameters:
        -----------
        session_id : str or int
            Session ID
        region : str or list
            Region acronym (e.g., 'RSC', 'SUB') or list of regions
            
        Returns:
        --------
        tuple
            (region_units_df, spike_times_dict)
        """
        # Convert session_id to int for AllenSDK
        session_id = int(session_id)
        
        # Get good units
        units = self.cache.get_unit_table()
        channels = self.cache.get_channel_table()
        good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]
        
        # Handle both string and list inputs for region
        if isinstance(region, list):
            region = region[0]  # Take first region if list is provided
        
        # Filter for target region and session
        region_units = good_units[
            (good_units.structure_acronym == region) & 
            (good_units.ecephys_session_id == session_id)
        ]
        
        try:
            # Get spike times for these units
            session = self.cache.get_ecephys_session(session_id)
            spike_times = {unit_id: session.spike_times[unit_id] for unit_id in region_units.index}
        except Exception as e:
            print(f"Error: {e}")
            print(f"region_units: {region_units.columns}")
            print(f"session_id: {session_id}")
            print(f"region: {region}")
            raise e
        
        return region_units, spike_times
        
    def calculate_firing_rate_changes(self, spike_times, swr_events, window_size=0.1):
        """
        Calculate firing rate changes around SWR events.
        
        Parameters:
        -----------
        spike_times : dict
            Dictionary mapping unit IDs to spike time arrays
        swr_events : pd.DataFrame
            DataFrame containing SWR events with columns:
            - start_time: Event start time
            - end_time: Event end time
            - power_peak_time: Time of peak power
            - power_max_zscore: Maximum power z-score
        window_size : float
            Size of time windows in seconds (default: 0.1)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with firing rate change statistics for each unit
        """
        # Pre-allocate arrays for results
        n_units = len(spike_times)
        n_events = len(swr_events)
        
        # Create arrays to store rates for each unit and event
        pre_event_rates = np.zeros((n_units, n_events))
        post_event_rates = np.zeros((n_units, n_events))
        before_peak_rates = np.zeros((n_units, n_events))
        after_peak_rates = np.zeros((n_units, n_events))
        during_rates = np.zeros((n_units, n_events))
        baseline_rates = np.zeros((n_units, n_events))
        
        # Calculate baseline from times not within 1s of any SWR
        all_swr_times = np.array([(event['start_time'] - 1, event['end_time'] + 1) 
                                 for _, event in swr_events.iterrows()])
        
        # Process each unit
        for i, (unit_id, times) in enumerate(spike_times.items()):
            # Convert times to numpy array for faster operations
            times = np.array(times)
            
            # Process each event
            for j, (_, event) in enumerate(swr_events.iterrows()):
                # Pre-event window
                pre_event_mask = (times >= event['start_time'] - window_size) & (times < event['start_time'])
                pre_event_rates[i, j] = np.sum(pre_event_mask) / window_size
                
                # Post-event window
                post_event_mask = (times > event['end_time']) & (times <= event['end_time'] + window_size)
                post_event_rates[i, j] = np.sum(post_event_mask) / window_size
                
                # Before peak window
                before_peak_mask = (times >= event['start_time']) & (times < event['power_peak_time'])
                before_peak_rates[i, j] = np.sum(before_peak_mask) / (event['power_peak_time'] - event['start_time'])
                
                # After peak window
                after_peak_mask = (times > event['power_peak_time']) & (times <= event['end_time'])
                after_peak_rates[i, j] = np.sum(after_peak_mask) / (event['end_time'] - event['power_peak_time'])
                
                # During event
                during_mask = (times >= event['start_time']) & (times <= event['end_time'])
                during_rates[i, j] = np.sum(during_mask) / (event['end_time'] - event['start_time'])
                
                # Baseline (times not within 1s of any SWR)
                baseline_mask = np.ones_like(times, dtype=bool)
                for swr_start, swr_end in all_swr_times:
                    baseline_mask &= ~((times >= swr_start) & (times <= swr_end))
                baseline_rates[i, j] = np.sum(baseline_mask) / np.sum(baseline_mask) if np.any(baseline_mask) else 0
        
        # Calculate statistics using vectorized operations
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        
        # Pre vs Post event
        t_stats, p_vals = stats.ttest_rel(post_event_rates, pre_event_rates, axis=1)
        pre_vs_post = [(np.mean(post_event_rates[i] - pre_event_rates[i]), p_vals[i]) for i in range(n_units)]
        
        # Before vs After peak
        t_stats, p_vals = stats.ttest_rel(after_peak_rates, before_peak_rates, axis=1)
        before_vs_after = [(np.mean(after_peak_rates[i] - before_peak_rates[i]), p_vals[i]) for i in range(n_units)]
        
        # During vs Baseline
        t_stats, p_vals = stats.ttest_rel(during_rates, baseline_rates, axis=1)
        during_vs_baseline = [(np.mean(during_rates[i] - baseline_rates[i]), p_vals[i]) for i in range(n_units)]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'unit_id': list(spike_times.keys()),
            'pre_vs_post': pre_vs_post,
            'before_vs_after': before_vs_after,
            'during_vs_baseline': during_vs_baseline
        })
        
        # Apply Benjamini-Hochberg correction
        for test in ['pre_vs_post', 'before_vs_after', 'during_vs_baseline']:
            # Extract p-values and handle NaN values
            p_values = np.array([x[1] for x in results[test]])
            valid_mask = ~np.isnan(p_values)
            
            if np.any(valid_mask):
                # Only correct valid p-values
                _, p_adjusted, _, _ = multipletests(
                    p_values[valid_mask], 
                    method='fdr_bh'
                )
                
                # Create corrected results with NaN for invalid p-values
                corrected_results = []
                valid_idx = 0
                for i, p_val in enumerate(p_values):
                    if np.isnan(p_val):
                        corrected_results.append((results[test][i][0], np.nan))
                    else:
                        corrected_results.append((results[test][i][0], p_adjusted[valid_idx]))
                        valid_idx += 1
                
                results[f'{test}_corrected'] = corrected_results
            else:
                # If no valid p-values, set all to NaN
                results[f'{test}_corrected'] = [(x[0], np.nan) for x in results[test]]
        
        return results

    def summarize_results(self, results_df, direction='any', p_threshold=0.05, use_corrected=True):
        """
        Summarize the results by counting significant units for each comparison.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Results DataFrame from calculate_firing_rate_changes
        direction : str
            'increase', 'decrease', or 'any' to specify the direction of change
        p_threshold : float
            P-value threshold for significance (default: 0.05)
        use_corrected : bool
            Whether to use corrected p-values (default: True)
            
        Returns:
        --------
        pd.DataFrame
            Summary of significant units for each comparison
        """
        summary = {}
        
        # Define the comparisons to analyze
        comparisons = ['pre_vs_post', 'before_vs_after', 'during_vs_baseline']
        
        for comp in comparisons:
            # Get the appropriate column based on whether to use corrected p-values
            p_col = f'{comp}_corrected' if use_corrected else comp
            
            # Extract effect sizes and p-values
            effects = [x[0] for x in results_df[p_col]]
            p_vals = [x[1] for x in results_df[p_col]]
            
            # Count significant units based on direction
            if direction == 'increase':
                sig_count = sum((np.array(effects) > 0) & (np.array(p_vals) < p_threshold))
            elif direction == 'decrease':
                sig_count = sum((np.array(effects) < 0) & (np.array(p_vals) < p_threshold))
            else:  # 'any'
                sig_count = sum(np.array(p_vals) < p_threshold)
            
            summary[comp] = {
                'significant_units': sig_count,
                'total_units': len(results_df),
                'percent_significant': (sig_count / len(results_df)) * 100
            }
        
        return pd.DataFrame(summary).T

    def analyze_session_swr_spikes(self, session_id, regions, window_size=0.05):
        """
        Analyze firing rate changes around SWR events for multiple regions in a session.
        
        Parameters:
        -----------
        session_id : str or int
            Session ID
        regions : list
            List of region acronyms to analyze
        window_size : float
            Size of time windows in seconds
            
        Returns:
        --------
        dict
            Dictionary mapping regions to analysis results DataFrames
        """
        results = {}
        
        # Convert session_id to string for file paths
        session_id_str = str(session_id)
        
        # Find all probe event files for this session
        session_path = os.path.join(self.swr_input_dir, self.dataset_name, f"swrs_session_{session_id_str}")
        if not os.path.exists(session_path):
            print(f"Session directory not found: {session_path}")
            return results
            
        # Find all putative SWR event files
        event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
        if not event_files:
            print(f"No SWR event files found in {session_path}")
            return results
            
        # Extract probe IDs from filenames using regex compilation for better performance
        probe_pattern = re.compile(r'probe_([^_]+)_channel_')
        probe_ids = set()
        for event_file in event_files:
            match = probe_pattern.search(os.path.basename(event_file))
            if match:
                probe_ids.add(int(match.group(1)))  # Convert probe ID to int
        
        if not probe_ids:
            print(f"Could not extract probe IDs from event files in {session_path}")
            return results
            
        print(f"Found {len(probe_ids)} probes for session {session_id_str}")
        
        # Collect events from all probes
        all_events = []
        for probe_id in probe_ids:
            try:
                probe_events = self.explorer.find_best_events(
                    dataset=self.dataset_name,
                    session_id=session_id_str,  # Keep as string for SWRExplorer
                    probe_id=str(probe_id),     # Convert to string for SWRExplorer
                    min_sw_power=MIN_SW_POWER,
                    min_duration=MIN_DURATION,
                    max_duration=MAX_DURATION,
                    min_clcorr=0.8,
                    exclude_gamma=True,
                    exclude_movement=True
                )
                if len(probe_events) > 0:
                    all_events.append(probe_events)
                    print(f"Found {len(probe_events)} events for probe {probe_id}")
            except Exception as e:
                print(f"Error processing probe {probe_id}: {str(e)}")
                continue
        
        if not all_events:
            print(f"No valid events found for session {session_id_str}")
            return results
            
        # Combine events from all probes
        swr_events = pd.concat(all_events, ignore_index=True)
        print(f"Total events across all probes: {len(swr_events)}")
        
        # Pre-fetch all region units and spike times to avoid repeated cache access
        region_data = {}
        for region in regions:
            try:
                region_units, spike_times = self.get_region_units_and_spikes(session_id, region)
                if len(spike_times) > 0:
                    region_data[region] = (region_units, spike_times)
                else:
                    print(f"No units found in {region} for session {session_id_str}")
            except Exception as e:
                print(f"Error getting data for region {region}: {str(e)}")
                continue
            
        # Analyze each region using pre-fetched data
        for region, (region_units, spike_times) in region_data.items():
            try:
                results[region] = self.calculate_firing_rate_changes(spike_times, swr_events, window_size)
            except Exception as e:
                print(f"Error analyzing region {region}: {str(e)}")
                continue
            
        return results

    def rank_regions_by_response(self, session_id, regions, event_df, p_threshold=0.05, use_corrected=True):
        """
        Rank regions by their response to a specific SWR event.
        
        Parameters:
        -----------
        session_id : str or int
            Session ID
        regions : str or list
            Single region or list of regions to analyze
        event_df : pd.DataFrame
            DataFrame containing the SWR event
        p_threshold : float
            P-value threshold for significance
        use_corrected : bool
            Whether to use corrected p-values
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regions ranked by their response to the event
        """
        # Convert single region to list
        if isinstance(regions, str):
            regions = [regions]
            
        region_scores = []
        
        for region in regions:
            try:
                # Get units and spike times
                region_units, spike_times = self.get_region_units_and_spikes(session_id, region)
                if len(spike_times) == 0:
                    continue
                    
                # Calculate firing rate changes
                fr_df = self.calculate_firing_rate_changes(spike_times, event_df)
                
                # Calculate response metrics
                effect_col = 'during_vs_baseline_corrected' if use_corrected else 'during_vs_baseline'
                effects = [x[0] for x in fr_df[effect_col]]
                pvals = [x[1] for x in fr_df[effect_col]]
                
                # Count significant units
                n_sig_units = sum((np.array(effects) > 0) & (np.array(pvals) < p_threshold))
                total_units = len(fr_df)
                percent_sig = (n_sig_units / total_units) * 100 if total_units > 0 else 0
                
                # Calculate mean effect size
                mean_effect = np.mean([e for e, p in zip(effects, pvals) if p < p_threshold]) if n_sig_units > 0 else 0
                
                region_scores.append({
                    'region': region,
                    'n_significant_units': n_sig_units,
                    'total_units': total_units,
                    'percent_significant': percent_sig,
                    'mean_effect_size': mean_effect
                })
                
            except Exception as e:
                print(f"Error processing region {region}: {str(e)}")
                continue
        
        # Create DataFrame and sort by response strength
        if region_scores:
            df = pd.DataFrame(region_scores)
            df['response_score'] = df['percent_significant'] * df['mean_effect_size']
            df = df.sort_values('response_score', ascending=False)
            return df
        else:
            return pd.DataFrame()

def find_best_spiking_coupling(
    cache_dir,
    swr_input_dir,
    output_dir,
    target_regions,
    min_units_per_region,
    window_size,
    save_intermediate_csv=False,
    save_regional_summary_csv=True
):
    """
    Find sessions with best spiking coupling to SWRs and analyze them.
    
    Parameters:
    -----------
    cache_dir : str
        Path to the AllenSDK cache directory
    swr_input_dir : str
        Path to the directory containing SWR event files
    output_dir : str
        Path to save output files
    target_regions : list
        List of brain regions to analyze
    min_units_per_region : int
        Minimum number of units required in each region
    window_size : float
        Size of analysis windows in seconds
    save_intermediate_csv : bool, default=False
        Whether to save intermediate results (unit-level data)
    save_regional_summary_csv : bool, default=True
        Whether to save the final regional summary CSV
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with significant unit counts for each session/region/direction
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SWRSpikeAnalyzer(
        cache_dir=cache_dir,
        swr_input_dir=swr_input_dir
    )
    
    # Get sessions with good unit counts
    search_df = abi_visual_behavior_units_session_search(
        cache_dir=cache_dir,
        target_regions=target_regions,
        min_units_per_region=min_units_per_region
    )
    
    # Add column to check if session has SWR data
    search_df['has_swr_data'] = search_df.index.map(
        lambda x: os.path.exists(os.path.join(swr_input_dir, "allen_visbehave_swr_murphylab2024", f"swrs_session_{x}"))
    )
    
    # Filter for sessions with SWR data
    search_df = search_df[search_df['has_swr_data']]
    
    if save_intermediate_csv:
        # Save the search results
        search_df.to_csv(os.path.join(output_dir, "abi_visual_behavior_units_session_search.csv"))
    
    # Initialize summary DataFrame
    summary_rows = []
    
    # Get the session with the most target region units for testing
    target_region = target_regions[0]  # Use first region as default
    if target_region in search_df.columns:
        test_session_id = search_df.index[0]  # First row has highest target_region count
        print(f"\nTesting with session {test_session_id} ({target_region} units: {search_df[target_region].iloc[0]})")
        
        # Analyze the test session
        results = analyzer.analyze_session_swr_spikes(
            session_id=test_session_id,
            regions=target_regions,
            window_size=window_size
        )
        
        # Save results for each region if requested
        if save_intermediate_csv:
            for region, region_results in results.items():
                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"swr_spike_analysis_{region}_{timestamp}.csv")
                region_results.to_csv(output_file, index=False)
                print(f"Saved results for {region} to {output_file}")
        
        # Generate summaries for different directions
        for region, region_results in results.items():
            for direction in ['increase', 'decrease', 'any']:
                # Get summary using corrected p-values
                corrected_summary = analyzer.summarize_results(
                    region_results, 
                    direction=direction,
                    p_threshold=0.05,
                    use_corrected=True
                )
                
                # Get summary using uncorrected p-values
                uncorrected_summary = analyzer.summarize_results(
                    region_results,
                    direction=direction,
                    p_threshold=0.05,
                    use_corrected=False
                )
                
                # Add to summary rows
                summary_rows.append({
                    'session_id': test_session_id,
                    'region': region,
                    'direction': direction,
                    'corrected_significant_units': corrected_summary['significant_units'].to_dict(),
                    'uncorrected_significant_units': uncorrected_summary['significant_units'].to_dict(),
                    'total_units': corrected_summary['total_units'].iloc[0]
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        
        if save_regional_summary_csv:
            # Save summary results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(output_dir, f"swr_spike_analysis_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSaved summary results to {summary_file}")
        
        return summary_df
    else:
        print(f"No sessions found with {target_region} units")
        return None

def suggest_best_events(analyzer, session_id, region, k=3, p_threshold=0.05, use_corrected=True):
    """
    Suggest the top k (probe, event) combos with the most units showing significant increase in firing rate.
    Returns a list of (probe_id, event_idx, n_significant_units).
    """
    # Find all probes for this session
    session_path = os.path.join(analyzer.swr_input_dir, analyzer.dataset_name, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    probe_ids = set()
    for event_file in event_files:
        match = re.search(r'probe_([^_]+)_channel_', os.path.basename(event_file))
        if match:
            probe_ids.add(match.group(1))
    results = []
    for probe_id in probe_ids:
        # Get events for this probe
        events = analyzer.explorer.find_best_events(
            dataset=analyzer.dataset_name,
            session_id=str(session_id),
            probe_id=str(probe_id),
            min_sw_power=1,
            min_duration=0.05,
            max_duration=0.15,
            min_clcorr=0,
            exclude_gamma=True,
            exclude_movement=True
        )
        if events.empty:
            continue
        # Get spikes for this region/probe
        region_units, spike_times = analyzer.get_region_units_and_spikes(session_id, region)
        if len(spike_times) == 0:
            continue
        # For each event, count significant units
        for idx, event in events.iterrows():
            event_df = pd.DataFrame([event])
            fr_df = analyzer.calculate_firing_rate_changes(spike_times, event_df)
            pvals = fr_df['during_vs_baseline_corrected' if use_corrected else 'during_vs_baseline']
            n_sig = sum([(x[0] > 0) and (x[1] < p_threshold) for x in pvals])
            results.append((probe_id, idx, n_sig))
    # Sort and return top k
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:k]

def spikes_around_event_plot(
    analyzer,
    session_id,
    probe_id,
    event_idx,
    regions,
    neuron_limits=None,
    fr_dfs=None,
    window=0.2,
    p_threshold=0.05,
    use_corrected=True,
    show_all_units=False,
    show_unit_ids=False,
    show_yticks=True,
    region_boundary_style=['black', '-', 1.0],  # [color, linestyle, linewidth]
    rank_regions=False  # New parameter to control region ranking
):
    """
    Plot a raster of spike times for units ranked by coupling to the SWR event.
    Spikes are shown as thin rectangles, and the highest-coupled units are at the top.
    
    Parameters:
    -----------
    analyzer : SWRSpikeAnalyzer
        The analyzer instance
    session_id : str or int
        Session ID
    probe_id : str or int
        Probe ID
    event_idx : int
        Index of the event to plot
    regions : str or list
        Single region or list of region acronyms to plot
    neuron_limits : list of int, optional
        List of maximum number of neurons to plot per region. Must match length of regions.
        If None, plots all neurons for each region.
    fr_dfs : dict, optional
        Dictionary mapping regions to firing rate DataFrames. If None, computes them.
    window : float
        Time window around event center in seconds
    p_threshold : float
        P-value threshold for significance
    use_corrected : bool
        Whether to use corrected p-values
    show_all_units : bool
        Whether to show all units or only significant ones
    show_unit_ids : bool
        Whether to show unit IDs on y-axis
    show_yticks : bool
        Whether to show y-axis ticks
    region_boundary_style : list
        [color, linestyle, linewidth] for region boundary lines
    rank_regions : bool
        Whether to rank regions by their response to the event
    """
    # Convert single region to list
    if isinstance(regions, str):
        regions = [regions]
        
    # Get events for this probe
    events = analyzer.explorer.find_best_events(
        dataset=analyzer.dataset_name,
        session_id=str(session_id),
        probe_id=str(probe_id),
        min_sw_power=1,
        min_duration=0.05,
        max_duration=0.15,
        min_clcorr=0.8,
        exclude_gamma=True,
        exclude_movement=True
    )
    event = events.loc[event_idx]
    t0 = event['start_time']
    t1 = event['end_time']
    t_peak = event['power_peak_time'] if 'power_peak_time' in event else (t0 + t1) / 2
    t_center = (t0 + t1) / 2
    
    # Create event DataFrame for ranking if needed
    event_df = pd.DataFrame([event])
    
    # Rank regions if requested
    if rank_regions:
        region_rankings = analyzer.rank_regions_by_response(
            session_id, regions, event_df, p_threshold, use_corrected
        )
        if not region_rankings.empty:
            print("\nRegion Rankings:")
            print(region_rankings[['region', 'n_significant_units', 'total_units', 'response_score']])
            # Reorder regions by ranking
            regions = region_rankings['region'].tolist()

    # Initialize plot data
    all_spike_xs = []
    all_spike_ys = []
    all_yticklabels = []
    region_boundaries = [0]  # Start with 0 for first region
    current_y = 0
    valid_regions = []

    # Process each region
    for i, region in enumerate(regions):
        try:
            # Get spikes for this region
            region_units, spike_times = analyzer.get_region_units_and_spikes(session_id, region)
            if len(spike_times) == 0:
                print(f"Warning: No units found in {region} for session {session_id}. Skipping.")
                continue

            # Get or compute firing rate DataFrame
            if fr_dfs is not None and region in fr_dfs:
                fr_df = fr_dfs[region]
            else:
                fr_df = analyzer.calculate_firing_rate_changes(spike_times, event_df)

            # Rank units by effect size
            effect_col = 'during_vs_baseline_corrected' if use_corrected else 'during_vs_baseline'
            fr_df['effect'] = [x[0] for x in fr_df[effect_col]]
            fr_df['pval'] = [x[1] for x in fr_df[effect_col]]
            fr_df = fr_df.sort_values('effect', ascending=False).reset_index(drop=True)

            # Select units to plot
            if show_all_units:
                plot_units = fr_df['unit_id'].values
            else:
                mask = (fr_df['effect'] > 0) & (fr_df['pval'] < p_threshold)
                plot_units = fr_df[mask]['unit_id'].values

            # Apply neuron limit if specified
            if neuron_limits is not None and i < len(neuron_limits):
                plot_units = plot_units[:neuron_limits[i]]

            # Add spikes for this region
            for rank, unit_id in enumerate(plot_units):
                spikes = spike_times[unit_id]
                rel_spikes = spikes[(spikes >= t_center - window/2) & (spikes <= t_center + window/2)] - t_center
                all_spike_xs.extend(rel_spikes)
                all_spike_ys.extend([current_y + rank] * len(rel_spikes))
                all_yticklabels.append(f"{region}-{unit_id}")

            # Update region boundary
            current_y += len(plot_units)
            region_boundaries.append(current_y)
            valid_regions.append(region)

        except Exception as e:
            print(f"Warning: Error processing region {region}: {str(e)}. Skipping.")
            continue

    if not valid_regions:
        print(f"Warning: No valid regions found to plot for session {session_id}. Skipping this session.")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot spikes as thin rectangles
    for x, y in zip(all_spike_xs, all_spike_ys):
        ax.vlines(x, y - 0.4, y + 0.4, color='red', linewidth=1)

    # Add region boundaries
    for boundary in region_boundaries[1:-1]:  # Skip first (0) and last (total height)
        ax.axhline(boundary - 0.5, color=region_boundary_style[0], 
                  linestyle=region_boundary_style[1], 
                  linewidth=region_boundary_style[2])

    # Add region labels
    for i, region in enumerate(valid_regions):
        y_pos = (region_boundaries[i] + region_boundaries[i+1]) / 2
        ax.text(-window/2 - 0.02, y_pos, region, 
                verticalalignment='center', horizontalalignment='right',
                fontweight='bold')

    # Mark event start/end/peak
    ax.plot([t0-t_center], [-1], marker='v', color='green', markersize=12, label='Event Start')
    ax.plot([t1-t_center], [-1], marker='v', color='green', markersize=12, label='Event End')
    ax.plot([t_peak-t_center], [-1], marker='v', color='black', markersize=12, label='Event Peak')

    # Only show one legend entry for start/end
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set axis properties
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron (ranked by coupling)')
    ax.set_title(f'Spikes for units (session {session_id}, probe {probe_id}, event {event_idx})')
    ax.set_ylim(-2, current_y + 2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    if show_unit_ids and show_yticks:
        ax.set_yticks(np.arange(len(all_yticklabels)))
        ax.set_yticklabels(all_yticklabels)
    elif not show_yticks:
        ax.set_yticks([])

    plt.tight_layout()
    return fig


