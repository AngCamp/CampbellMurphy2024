#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
import os
import sys

# Add parent directory to path to import SWRExplorer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from allensdk.brain_observatory.ecephys.visual_behavior_visualization import VisualBehaviorNeuropixelsProjectCache
import numpy as np
import pandas as pd

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
    channels = cache.get_channel_table()

    print("\nInitial data shapes:")
    print(f"Units table shape: {units.shape}")
    print(f"Channels table shape: {channels.shape}")

    # Filter for good units
    good_units = units[(units['quality'] == 'good') & (units['valid_data'] == True)]

    # Count good units per channel and join to channels
    good_unit_counts = good_units.groupby('ecephys_channel_id').size().rename('good_unit_count')
    channels = channels.join(good_unit_counts, how='left')
    channels['good_unit_count'] = channels['good_unit_count'].fillna(0).astype(int)

    # Filter for target regions with valid data
    region_mask = channels['structure_acronym'].isin(target_regions) & channels['valid_data']
    roi_channels = channels[region_mask].copy()

    # Group by session and region
    grouped_counts = (
        roi_channels
        .groupby(['ecephys_session_id', 'structure_acronym'])['good_unit_count']
        .sum()
        .unstack(fill_value=0)
    )

    # Identify sessions passing threshold
    session_pass_mask = (grouped_counts >= min_units_per_region).any(axis=1)
    passed_sessions_df = grouped_counts[session_pass_mask].copy()

    # Add total across all target regions
    passed_sessions_df['total_good_units_in_rois'] = passed_sessions_df.sum(axis=1)

    print(f"\nNumber of sessions passing threshold: {len(passed_sessions_df)}")
    return passed_sessions_df
