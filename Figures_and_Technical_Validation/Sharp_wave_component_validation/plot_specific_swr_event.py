#!/usr/bin/env python3
"""
Script to plot a specific SWR event from a given dataset/session/probe/event index,
using the modular plotting options in SWRExplorer.
All parameters are set as variables below for easy notebook/script use.
"""
import matplotlib.pyplot as plt
from SWRExplorer import SWRExplorer
import os

# --- User parameters (set these as needed) ---
dataset = "allen_visbehave_swr_murphylab2024"
session_id = "1093864136"  # Replace with your session ID
probe_id = "1094073091"    # Replace with your probe ID
event_id = 829         # Index in the probe's event CSV (not global)
image_type = 'svg'
filter_path = os.path.join(
    os.sep, "home", "acampbell", "NeuropixelsLFPOnRamp", "SWR_Neuropixels_Detector", "Filters", "sharpwave_componenet_8to40band_1500hz_band.npz"
)
envelope_mode = 'zscore'  # Options: 'zscore' or 'raw'

# Optional modular plotting options:
window_padding = 0.02  # seconds - controls how much time we see around the event
figsize_mm = (200, 200)  # Keep same proportions as original workflow
panels_to_plot = ['raw_pyramidal_lfp', 'raw_s_radiatum_lfp', 'bandpassed_signals', 'power']  # All four panels
time_per_mm = None     # Let figsize_mm control the width
make_one_plot = True # make one plot instead of one for each panel


# Output path for SVG
output_dir = os.path.join(
    os.sep, "home", "acampbell", "NeuropixelsLFPOnRamp", "Figures_and_Technical_Validation", "Sharp_wave_component_validation"
)

# --- Main logic ---
explorer = SWRExplorer(base_path=os.path.join(
    os.sep, "space", "scratch", "SWR_final_pipeline", "osf_campbellmurphy2025_swr_data"
))

# Fetch the event (by CSV index)
events_df = explorer.data[dataset][session_id][probe_id]['events']
# Add required metadata columns for plotting
for col, val in zip(['dataset', 'session_id', 'probe_id'], [dataset, session_id, probe_id]):
    events_df[col] = val

# Debug: Print event info for event_id and event_id-1
print(f"Event at event_id={event_id}:")
print(events_df.loc[event_id][['start_time', 'end_time', 'sw_peak_power']])
if event_id > 0:
    print(f"Event at event_id={event_id-1}:")
    print(events_df.loc[event_id-1][['start_time', 'end_time', 'sw_peak_power']])

# Set time_per_mm for proportional width, only use figsize_mm for height
# (e.g., 0.001 means 1 ms per mm)
time_per_mm = 0.001  # Adjust as needed for your preferred scaling
fig_height_mm = 80   # Only set height, width will be computed
figsize_mm = (None, fig_height_mm)

# Plot the event with modular options
# make one plot

for i, panel in enumerate(panels_to_plot):
    if make_one_plot:
        # set up output filename
        output_filename = os.path.join(
            output_dir, f"swr_event_{event_id}_session_{session_id}_probe_{probe_id}.{image_type}"
        )
        panel = panels_to_plot
    else:
        # set up output filename
        output_filename = os.path.join(
            output_dir, f"swr_event_{event_id}_session_{session_id}_probe_{probe_id}_{panel}.{image_type}"
        )
        panel = [panel]

    # plot the event - pass panel as a single-item list
    fig = explorer.plot_swr_event(
        events_df=events_df,
        event_idx=event_id,
        filter_path=filter_path,
        window_padding=window_padding,
        figsize_mm=figsize_mm,
        panels_to_plot=panel,  # Pass as a list containing just this panel
        time_per_mm=time_per_mm,
        dpi=300,  # Add high resolution
        envelope_mode=envelope_mode
    )
    # save the figure
    fig.savefig(output_filename, format=image_type, dpi=600, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    if make_one_plot:
        break
 