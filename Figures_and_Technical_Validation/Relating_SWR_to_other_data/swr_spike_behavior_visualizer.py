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
        region : str
            Region acronym (e.g., 'RSC', 'SUB')
            
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
        results = []
        
        # Calculate baseline from times not within 1s of any SWR
        all_swr_times = []
        for _, event in swr_events.iterrows():
            all_swr_times.append((event['start_time'] - 1, event['end_time'] + 1))
        
        for unit_id, times in spike_times.items():
            unit_results = {'unit_id': unit_id}
            
            # Calculate firing rates for different windows
            for _, event in swr_events.iterrows():
                # Pre-event window
                pre_event_mask = (times >= event['start_time'] - window_size) & (times < event['start_time'])
                pre_event_rate = np.sum(pre_event_mask) / window_size
                
                # Post-event window
                post_event_mask = (times > event['end_time']) & (times <= event['end_time'] + window_size)
                post_event_rate = np.sum(post_event_mask) / window_size
                
                # Before peak window
                before_peak_mask = (times >= event['start_time']) & (times < event['power_peak_time'])
                before_peak_rate = np.sum(before_peak_mask) / (event['power_peak_time'] - event['start_time'])
                
                # After peak window
                after_peak_mask = (times > event['power_peak_time']) & (times <= event['end_time'])
                after_peak_rate = np.sum(after_peak_mask) / (event['end_time'] - event['power_peak_time'])
                
                # During event
                during_mask = (times >= event['start_time']) & (times <= event['end_time'])
                during_rate = np.sum(during_mask) / (event['end_time'] - event['start_time'])
                
                # Baseline (times not within 1s of any SWR)
                baseline_mask = np.ones_like(times, dtype=bool)
                for swr_start, swr_end in all_swr_times:
                    baseline_mask &= ~((times >= swr_start) & (times <= swr_end))
                baseline_rate = np.sum(baseline_mask) / np.sum(baseline_mask) if np.any(baseline_mask) else 0
                
                # Store rates
                unit_results.setdefault('pre_event_rates', []).append(pre_event_rate)
                unit_results.setdefault('post_event_rates', []).append(post_event_rate)
                unit_results.setdefault('before_peak_rates', []).append(before_peak_rate)
                unit_results.setdefault('after_peak_rates', []).append(after_peak_rate)
                unit_results.setdefault('during_rates', []).append(during_rate)
                unit_results.setdefault('baseline_rates', []).append(baseline_rate)
            
            # Calculate t-tests
            from scipy import stats
            from statsmodels.stats.multitest import multipletests
            
            # Pre vs Post event
            t_stat, p_val = stats.ttest_rel(unit_results['pre_event_rates'], unit_results['post_event_rates'])
            unit_results['pre_vs_post'] = (np.mean(unit_results['post_event_rates']) - np.mean(unit_results['pre_event_rates']), p_val)
            
            # Before vs After peak
            t_stat, p_val = stats.ttest_rel(unit_results['before_peak_rates'], unit_results['after_peak_rates'])
            unit_results['before_vs_after'] = (np.mean(unit_results['after_peak_rates']) - np.mean(unit_results['before_peak_rates']), p_val)
            
            # During vs Baseline
            t_stat, p_val = stats.ttest_rel(unit_results['during_rates'], unit_results['baseline_rates'])
            unit_results['during_vs_baseline'] = (np.mean(unit_results['during_rates']) - np.mean(unit_results['baseline_rates']), p_val)
            
            results.append(unit_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply Benjamini-Hochberg correction
        for test in ['pre_vs_post', 'before_vs_after', 'during_vs_baseline']:
            # Extract p-values and handle NaN values
            p_values = [x[1] for x in results_df[test]]
            valid_mask = ~np.isnan(p_values)
            
            if np.any(valid_mask):
                # Only correct valid p-values
                _, p_adjusted, _, _ = multipletests(
                    np.array(p_values)[valid_mask], 
                    method='fdr_bh'
                )
                
                # Create corrected results with NaN for invalid p-values
                corrected_results = []
                valid_idx = 0
                for i, p_val in enumerate(p_values):
                    if np.isnan(p_val):
                        corrected_results.append((results_df[test][i][0], np.nan))
                    else:
                        corrected_results.append((results_df[test][i][0], p_adjusted[valid_idx]))
                        valid_idx += 1
                
                results_df[f'{test}_corrected'] = corrected_results
            else:
                # If no valid p-values, set all to NaN
                results_df[f'{test}_corrected'] = [(x[0], np.nan) for x in results_df[test]]
        
        return results_df

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
            
        # Extract probe IDs from filenames
        probe_ids = set()
        for event_file in event_files:
            match = re.search(r'probe_([^_]+)_channel_', os.path.basename(event_file))
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
            
        # Analyze each region
        for region in regions:
            region_units, spike_times = self.get_region_units_and_spikes(session_id, region)
            if len(spike_times) == 0:
                print(f"No units found in {region} for session {session_id_str}")
                continue
                
            results[region] = self.calculate_firing_rate_changes(spike_times, swr_events, window_size)
            
        return results

def find_best_spiking_coupling(
    cache_dir,
    swr_input_dir,
    output_dir,
    target_regions,
    min_units_per_region,
    window_size,
    save_intermediate=True
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
    save_intermediate : bool
        Whether to save intermediate results (unit-level data)
        
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
    
    if save_intermediate:
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
        if save_intermediate:
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
        
        if save_intermediate:
            # Save summary results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(output_dir, f"swr_spike_analysis_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSaved summary results to {summary_file}")
        
        return summary_df
    else:
        print(f"No sessions found with {target_region} units")
        return None


