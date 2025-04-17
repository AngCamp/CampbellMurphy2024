# Guide to Update SW_metric_EDA.ipynb

This document provides guidance and code snippets to update the `SW_metric_EDA.ipynb` notebook to work with the refactored `SWR_Neuropixels_Detector` pipeline structure and its outputs.

## Summary of Pipeline Changes Relevant to the Notebook

The main pipeline (`SWR_Neuropixels_Detector`) has undergone several changes affecting how data is stored and accessed:

1.  **Base Path:** The project root is now `SWR_Neuropixels_Detector/`. Outputs are typically located in a subdirectory specified in the `united_detector_config.yaml` (e.g., `./swr_output/` by default relative to the project root).
2.  **Output Structure:** Outputs are organized within session-specific subfolders: `OUTPUT_BASE_DIR/swrs_session_<session_id>/`.
3.  **Aggregated Probe Metadata:** General information about all probes within a session (channel counts, unit counts, position ranges) is now stored in a single CSV file: `OUTPUT_BASE_DIR/swrs_session_<session_id>/probe_metadata.csv.gz`.
4.  **Channel Selection Metadata:** Detailed metrics used during the ripple and sharp-wave channel selection process for *each individual probe* are now stored in separate JSON files: `OUTPUT_BASE_DIR/swrs_session_<session_id>/probe_<probe_id>_channel_selection_metadata.json.gz`. 
    *   **Note:** This file is only created if the pipeline was run with the `-m` or `--save-channel-metadata` flag.
    *   This file contains detailed lists of metrics (e.g., modulation index, circular-linear correlation, net power) for all CA1 channels evaluated during the sharp-wave channel selection process.
5.  **SWR Event Files:** Putative/filtered SWR events for each probe are stored in: `OUTPUT_BASE_DIR/swrs_session_<session_id>/probe_<probe_id>_karlsson_detector_events.csv.gz`.
    *   **Note:** If the filtering stage (`-f`) was run, this file is **overwritten** and now contains additional columns related to artifact overlap (e.g., `gamma_overlap_percent`, `movement_overlap_percent`) and sharp-wave component metrics calculated *per event* (e.g., `sw_exceeds_threshold`, `sw_peak_power`, `sw_peak_time`, `sw_ripple_plv`, `sw_ripple_mi`, `sw_ripple_clcorr`).

## Updating the Notebook Code

You need to replace the old data loading logic in `SW_metric_EDA.ipynb` with code that accesses these new file paths and formats. Below are example Python snippets.

**Instructions:**

1.  Adapt the `output_base_dir`, `session_id`, and `probe_id` variables in the snippets to match your analysis needs.
2.  Integrate these snippets into the appropriate cells of your notebook.
3.  Modify the subsequent analysis code in the notebook to use the new variable names (e.g., `all_probes_df`, `channel_selection_metadata`, `swr_events_df`) and the data structures they contain (DataFrames, dictionaries). Pay close attention to the new columns available in `swr_events_df` if the filtering stage was run.

```python
import pandas as pd
import json
import gzip
import os
import matplotlib.pyplot as plt # Make sure plotting libraries are imported
import numpy as np

# --- Configuration (Adapt these paths) ---
# Base directory where the pipeline outputs are stored
# Adjust this path relative to the notebook's location or use an absolute path
output_base_dir = '../../SWR_Neuropixels_Detector/swr_output' # Example relative path

# Example session ID you want to analyze
session_id = 'YOUR_SESSION_ID' # !!! Replace with an actual session ID !!!

# Example probe ID for session-probe specific files
probe_id = 'YOUR_PROBE_ID' # !!! Replace with an actual probe ID for the session !!!

session_folder = os.path.join(output_base_dir, f'swrs_session_{session_id}')
print(f"Looking for data in: {session_folder}")

# --- Loading Aggregated Probe Metadata ---
probe_metadata_path = os.path.join(session_folder, 'probe_metadata.csv.gz')
all_probes_df = None
try:
    with gzip.open(probe_metadata_path, 'rt') as f:
        all_probes_df = pd.read_csv(f)
    print(f"Loaded aggregated probe metadata from: {probe_metadata_path}")
    # print(all_probes_df.head())
except FileNotFoundError:
    print(f"ERROR: Aggregated probe metadata file not found: {probe_metadata_path}")
except Exception as e:
    print(f"ERROR loading aggregated probe metadata: {e}")

# --- Loading Channel Selection Metadata (for a specific probe) ---
# Note: This file might not exist if --save-channel-metadata was not used during the pipeline run
chan_sel_meta_path = os.path.join(session_folder, f'probe_{probe_id}_channel_selection_metadata.json.gz')
channel_selection_metadata = None
try:
    with gzip.open(chan_sel_meta_path, 'rt', encoding='utf-8') as f:
        channel_selection_metadata = json.load(f)
    print(f"Loaded channel selection metadata for probe {probe_id} from: {chan_sel_meta_path}")
    # Example access:
    # selected_ripple_chan = channel_selection_metadata.get('selected_ripple_channel_id')
    # selected_sw_chan = channel_selection_metadata.get('selected_sharpwave_channel_id')
    # sw_metrics_dict = channel_selection_metadata.get('sharpwave_selection_metrics', {})
    # print(f" Selected Ripple Chan: {selected_ripple_chan}, SW Chan: {selected_sw_chan}")
    # print(f" SW Metrics Keys: {list(sw_metrics_dict.keys())}") # E.g., ['channel_ids', 'depths', 'modulation_index', 'circulinear_correlation', 'net_sw_power']
except FileNotFoundError:
    print(f"WARNING: Channel selection metadata file not found (was --save-channel-metadata used?): {chan_sel_meta_path}")
except Exception as e:
    print(f"ERROR loading channel selection metadata for probe {probe_id}: {e}")

# --- Loading Filtered SWR Events (for a specific probe) ---
events_path = os.path.join(session_folder, f'probe_{probe_id}_karlsson_detector_events.csv.gz')
swr_events_df = None
try:
    with gzip.open(events_path, 'rt') as f:
        swr_events_df = pd.read_csv(f)
    print(f"Loaded SWR events for probe {probe_id} from: {events_path}")
    # Expect columns like: start_time, end_time, duration, Peak_time,
    # overlaps_with_gamma, gamma_overlap_percent,
    # overlaps_with_movement, movement_overlap_percent,
    # sw_exceeds_threshold, sw_peak_power, sw_peak_time,
    # sw_ripple_plv, sw_ripple_mi, sw_ripple_clcorr (if filtering stage ran)
    # print(swr_events_df.head())
    # print(list(swr_events_df.columns))
except FileNotFoundError:
    print(f"ERROR: SWR events file not found: {events_path}")
except Exception as e:
    print(f"ERROR loading SWR events for probe {probe_id}: {e}")

# --- Example EDA using the loaded data ---

# Analyze SW metrics from the channel selection process (if loaded)
if channel_selection_metadata:
    sw_selection_metrics_dict = channel_selection_metadata.get('sharpwave_selection_metrics', {})
    if sw_selection_metrics_dict and all(k in sw_selection_metrics_dict for k in ['channel_ids', 'modulation_index', 'circulinear_correlation', 'net_sw_power']):
        try:
            sw_metrics_df = pd.DataFrame(sw_selection_metrics_dict)
            print("\n--- SW Channel Selection Metrics (All Evaluated Channels Below Ripple Channel) ---")
            print(sw_metrics_df[['modulation_index', 'circulinear_correlation', 'net_sw_power']].describe())
            
            # Example Plot: Histogram of Modulation Index values during selection
            plt.figure()
            sw_metrics_df['modulation_index'].dropna().hist(bins=15)
            plt.title(f'Probe {probe_id}: Distribution of SW-Ripple MI (Channel Selection)')
            plt.xlabel('Modulation Index (MI)')
            plt.ylabel('Channel Count')
            plt.show()
        except Exception as plot_e:
            print(f"Error plotting SW selection metrics: {plot_e}")
    else:
        print("\nSW selection metrics dictionary is missing required keys.")
else:
    print("\nChannel selection metadata not loaded, skipping related EDA.")

# Analyze per-event SW metrics added during filtering stage (if loaded)
if swr_events_df is not None:
    sw_metric_cols = ['sw_peak_power', 'sw_ripple_mi', 'sw_ripple_clcorr', 'sw_ripple_plv']
    if all(col in swr_events_df.columns for col in sw_metric_cols):
        print("\n--- Per-Event SW Metrics (Filtered Events) ---")
        print(swr_events_df[sw_metric_cols].describe())
        
        # Example Plot: Scatter plot of per-event MI vs. SW Peak Power
        plt.figure()
        plt.scatter(swr_events_df['sw_peak_power'], swr_events_df['sw_ripple_mi'], alpha=0.5, s=10)
        plt.title(f'Probe {probe_id}: Per-Event SW-Ripple MI vs. SW Power')
        plt.xlabel('SW Peak Power (z-score)')
        plt.ylabel('Modulation Index (MI)')
        plt.grid(True)
        plt.show()
    else:
        print("\nSWR events DataFrame does not contain expected per-event SW metric columns (was filter stage run?).")
else:
    print("\nSWR events DataFrame not loaded, skipping related EDA.")

print("\n--- Update Guide Complete ---") 