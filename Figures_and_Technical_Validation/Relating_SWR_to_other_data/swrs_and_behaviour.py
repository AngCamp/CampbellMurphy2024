import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import scipy.ndimage

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

# --- CONFIG ---
CACHE_DIR = "/space/scratch/allen_visbehave_data"
SWR_INPUT_DIR = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
DATASET_NAME = "allen_visbehave_swr_murphylab2024"
SESSION_ID = 1047969464  # Example session
WINDOW = 2.5  # seconds before and after transition
MAX_PLOTS = 10
OUTPUT_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_and_Technical_Validation/Relating_SWR_to_other_data/swrs_pupils_and_running_results"
SWR_COUNT_THRESHOLD = 1  # Minimum number of SWRs in window to plot

# --- Helper: Load and filter SWR events ---
def load_and_filter_swr_events(session_id):
    session_path = os.path.join(SWR_INPUT_DIR, DATASET_NAME, f"swrs_session_{session_id}")
    event_files = glob.glob(os.path.join(session_path, "probe_*_channel_*_putative_swr_events.csv.gz"))
    all_events = []
    for event_file in event_files:
        try:
            events_df = pd.read_csv(event_file, compression='gzip')
            if not events_df.empty:
                all_events.append(events_df)
        except Exception as e:
            print(f"Warning: Could not load {event_file}: {e}")
            continue
    if not all_events:
        return pd.DataFrame([])
    events = pd.concat(all_events, ignore_index=True)
    # Filter as specified
    filtered = events[
        (events['sw_peak_power'] > 1) &
        (events['power_max_zscore'] > 3) & (events['power_max_zscore'] < 10) &
        (events['overlaps_with_gamma'] == True) &
        (events['overlaps_with_movement'] == True)
    ].copy()
    return filtered

# --- Helper: Find running speed transitions ---
def find_speed_transitions(running_speed, threshold=2.0):
    # running_speed: DataFrame with 'timestamps' and 'speed'
    speed = running_speed['speed'].values
    times = running_speed['timestamps'].values
    # Upward: from <2 to >=2
    up_idx = np.where((speed[:-1] < threshold) & (speed[1:] >= threshold))[0] + 1
    # Downward: from >=2 to <2
    down_idx = np.where((speed[:-1] >= threshold) & (speed[1:] < threshold))[0] + 1
    up_times = times[up_idx]
    down_times = times[down_idx]
    return up_times, down_times

# --- Main plotting function ---
def plot_transitions_with_swrs(session, running_speed, eye_tracking, swr_events, up_times, down_times, window=2.5, max_plots=10, swr_count_threshold=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_made = 0
    all_transitions = np.concatenate([
        np.array([(t, 'up') for t in up_times]),
        np.array([(t, 'down') for t in down_times])
    ], dtype=object)
    # Sort by time
    all_transitions = all_transitions[np.argsort(all_transitions[:,0].astype(float))]
    for t, direction in all_transitions:
        t = float(t)
        # Find SWR events in window (include any overlapping SWR)
        swrs_in_window = swr_events[(swr_events['end_time'] >= t-window) & (swr_events['start_time'] <= t+window)]
        n_swrs = len(swrs_in_window)
        if n_swrs < swr_count_threshold:
            print(f"Transition at {t:.2f}s ({direction}): {n_swrs} SWRs in window (below threshold {swr_count_threshold}, skipping)")
            continue
        else:
            print(f"Transition at {t:.2f}s ({direction}): {n_swrs} SWRs in window (meets threshold {swr_count_threshold}, plotting)")
        # Get running and pupil data in window
        trial_running = running_speed.query('timestamps >= @t-@window and timestamps <= @t+@window').copy()
        trial_pupil_area = eye_tracking.query('timestamps >= @t-@window and timestamps <= @t+@window').copy()
        # Only plot where both running and pupil data exist
        min_time = max(trial_running['timestamps'].min(), trial_pupil_area['timestamps'].min())
        max_time = min(trial_running['timestamps'].max(), trial_pupil_area['timestamps'].max())
        trial_running = trial_running[(trial_running['timestamps'] >= min_time) & (trial_running['timestamps'] <= max_time)]
        trial_pupil_area = trial_pupil_area[(trial_pupil_area['timestamps'] >= min_time) & (trial_pupil_area['timestamps'] <= max_time)]
        # Smooth pupil area with half-Gaussian (causal)
        pupil_vals = trial_pupil_area['pupil_area'].values
        pupil_times = trial_pupil_area['timestamps'].values
        if len(pupil_vals) > 1:
            # Calculate sigma in samples
            time_diffs = np.diff(pupil_times)
            median_dt = np.median(time_diffs)
            sigma_samples = max(1, int(0.004 / median_dt))
            # Create half-Gaussian kernel (causal)
            kernel_len = sigma_samples * 6
            x = np.arange(0, kernel_len)
            kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
            kernel = kernel / kernel.sum()
            # Use origin=0 for causal smoothing
            pupil_smooth = scipy.ndimage.convolve1d(pupil_vals, kernel, mode='nearest', origin=0)
            # Trim to match original length if needed
            if len(pupil_smooth) > len(pupil_vals):
                pupil_smooth = pupil_smooth[:len(pupil_vals)]
            elif len(pupil_smooth) < len(pupil_vals):
                pupil_smooth = np.pad(pupil_smooth, (0, len(pupil_vals)-len(pupil_smooth)), mode='edge')
            pupil_z = (pupil_smooth - np.nanmean(pupil_smooth)) / np.nanstd(pupil_smooth)
        else:
            pupil_z = pupil_vals
        # Absolute value for running speed, no negatives
        speed_vals = np.abs(trial_running['speed'].values)
        # Plot
        fig, axr = plt.subplots(figsize=(10,5))
        axr.margins(x=0, y=0)
        axp = axr.twinx()
        # Plot running speed (orange)
        axr.plot(trial_running['timestamps'], speed_vals, color='orange', label='Running speed', linewidth=2, zorder=1)
        # Plot pupil area (smoothed, z-scored, green)
        axp.plot(trial_pupil_area['timestamps'], pupil_z, color='green', label='Pupil (z)', linewidth=2, zorder=1)
        # Remove y-axis margins
        axr.set_ylim(bottom=0)
        axp.margins(y=0)
        # Mark SWR events in window (plot on top)
        for _, event in swrs_in_window.iterrows():
            # Green span
            axr.axvspan(event['start_time'], event['end_time'], color='green', alpha=0.2, zorder=5)
            # Black dotted line at peak
            if not np.isnan(event['power_peak_time']):
                axr.axvline(event['power_peak_time'], color='black', linestyle=':', linewidth=2, zorder=10)
                # Triangle marker at peak
                ymin, ymax = axr.get_ylim()
                axr.plot(event['power_peak_time'], ymin+0.05*(ymax-ymin), marker='v', color='black', markersize=12, zorder=10)
        # Styling
        axr.set_ylabel('Running speed (cm/s)', fontsize=14, fontweight='bold')
        axp.set_ylabel('Pupil area (z-score)', fontsize=14, fontweight='bold')
        axr.set_xlabel('Experiment time (s)', fontsize=14, fontweight='bold')
        axr.set_title(f"Session {SESSION_ID} | {direction} transition at {t:.2f}s | {n_swrs} SWRs in window", fontsize=14, fontweight='bold')
        for label in (axr.get_xticklabels() + axr.get_yticklabels() + axp.get_yticklabels()):
            label.set_fontsize(14)
            label.set_fontweight('bold')
        # Remove plot margins
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.15)
        plt.tight_layout()
        # Save plot as PNG and SVG
        fname_base = f"swrs_pupil_running_session{SESSION_ID}_{direction}_transition_{t:.2f}s"
        fpath_png = os.path.join(OUTPUT_DIR, fname_base + ".png")
        fpath_svg = os.path.join(OUTPUT_DIR, fname_base + ".svg")
        fig.savefig(fpath_png)
        fig.savefig(fpath_svg)
        plt.close(fig)
        plots_made += 1
        if plots_made >= max_plots:
            break

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Load session
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=CACHE_DIR)
    session = cache.get_ecephys_session(SESSION_ID)
    running_speed = session.running_speed
    eye_tracking = session.eye_tracking
    # Remove blinks if possible (if column exists)
    if 'blink' in eye_tracking.columns:
        eye_tracking_noblinks = eye_tracking[~eye_tracking['blink']]
    else:
        eye_tracking_noblinks = eye_tracking
    # Load and filter SWR events
    swr_events = load_and_filter_swr_events(SESSION_ID)
    print(f"Total filtered SWR events: {len(swr_events)}")
    # Find running speed transitions
    up_times, down_times = find_speed_transitions(running_speed)
    # Plot
    plot_transitions_with_swrs(session, running_speed, eye_tracking_noblinks, swr_events, up_times, down_times, window=WINDOW, max_plots=MAX_PLOTS, swr_count_threshold=SWR_COUNT_THRESHOLD)