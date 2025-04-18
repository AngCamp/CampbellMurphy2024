# libraries
import os
import re
import subprocess
import numpy as np
import pandas as pd
from scipy import io, signal

# from fitter import Fitter, get_common_distributions, get_distributions
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# for ripple detection
import ripple_detection
import ripple_detection.simulate as ripsim  # for making our time vectors
from scipy import signal
import seaborn as sns

# import KernelRegDraft as kreg # custom module, not needed
# import 'Stienmetz2019Reanalyzed/KernelRegDraft.py' as kreg
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from tqdm import tqdm
import yaml
import traceback

# Load the configuration from a YAML file
with open("ibl_swr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get the values from the configuration
input_dir_global = config["input_dir_global"]
global_rip_label = config["global_rip_label"]
minimum_ripple_num = config["minimum_ripple_num"]
merge_events_offset = config["merge_events_offset"]
no_gamma = config["global_no_gamma"]


# Functions
def check_overlap(df1, df2, offset=0):
    # returns true or false if there is overlap between the two dataframes with start_time and end_time columns
    result = []

    for i in range(len(df1)):
        start_time_1, end_time_1 = (
            df1.loc[i, "start_time"] - offset,
            df1.loc[i, "end_time"] + offset,
        )
        overlap = any(
            (start_time_2 < end_time_1 and end_time_2 > start_time_1)
            for start_time_2, end_time_2 in zip(df2["start_time"], df2["end_time"])
        )
        result.append(overlap)

    return result


def merge_intervals(intervals, x_offset=0):
    """
    Merges overlapping intervals within a list of intervals
    Convert intervals into sets of tuples then merge with this function

    parameters
    intervals : list of tuples
    x_offset : int, optional, time offset to merge intervals

    Returns a list of merged intervals
    """
    intervals.sort(key=lambda interval: interval[0])

    stack = [intervals[0]]
    for current in intervals[1:]:
        last = stack[-1]
        if current[0] <= last[1] + x_offset:
            stack[-1] = (last[0], max(last[1], current[1]))
        else:
            stack.append(current)

    return stack


# Function to convert DataFrame to set of tuples (start_time, end_time)
def dataframe_to_interval_set(df):
    interval_set = set()
    for index, row in df.iterrows():
        interval_set.add((row["start_time"], row["end_time"]))
    return interval_set


def union_of_lists_of_interval_sets(list_of_interval_sets):
    """
    Takes a list of lists of interval sets and returns the union of all
    interval sets in the list

    parameters
    list_of_interval_sets : list of lists of interval sets

    Returns a list of merged intervals
    """
    union = set()
    for interval_set in list_of_interval_sets:
        union = union.union(interval_set)
    return union


def union_of_event_times(list_of_event_intervals_df, offset=0):
    """
    Returns the union of all event times in a list of DataFrames
    Parameters
    ----------
    list_of_event_intervals_df : list of DataFrames
        Each DataFrame should have columns 'start_time' and 'end_time'
    offset : int, optional, time offset to merge intervals
    Returns
    -------
    list : list of tuples
        Each tuple is an event time interval
    """
    set_list = []
    for df in list_of_event_intervals_df:
        set_list.append(dataframe_to_interval_set(df))

    union = union_of_lists_of_interval_sets(set_list)

    return merge_intervals(list(union), x_offset=offset)


def dataframe_union_of_event_times(
    list_of_event_intervals_df, offset=0, event_name="event"
):
    """
    Returns the union of all event times in a list of DataFrames
    Parameters
    ----------
    list_of_event_intervals_df : list of DataFrames
        Each DataFrame should have columns 'start_time' and 'end_time'
    offset : int, optional, time offset to merge intervals
    Returns
    -------
    DataFrame : DataFrame of event times
        Columns are 'start_time' and 'end_time'
    """
    union = union_of_event_times(list_of_event_intervals_df, offset=offset)
    union_df = pd.DataFrame(union, columns=["start_time", "end_time"])
    union_df[event_name] = union_df.index
    return union_df


def add_overlap_probes(df, df_dict, overlap_col_name="overlap_probes", offset=0):
    overlap_probes = []
    overlap_probes_row = []
    peak_times = []
    max_zscores = []
    peak_probes = []
    for i, row in df.iterrows():
        overlap_probes_this_event = []
        overlap_probes_row_this_event = []
        peak_times_row = []
        max_zscores_row = []
        peak_probes_row = []
        for key, df_probe in df_dict.items():
            overlap = (df_probe["start_time"] < row["end_time"] + offset) & (
                df_probe["end_time"] > row["start_time"] - offset
            )
            if any(overlap):
                overlap_probes_this_event.append(key)
                peak_times_row.append(df_probe.loc[overlap, "Peak_time"].values[0])
                max_zscores_row.append(df_probe.loc[overlap, "max_zscore"].values[0])
                peak_probes_row.append(key)
                overlap_probes_row_this_event.extend(np.where(overlap)[0].tolist())
        overlap_probes.append(overlap_probes_this_event)
        overlap_probes_row.append(overlap_probes_row_this_event)
        max_zscore_idx = max_zscores_row.index(max(max_zscores_row))
        peak_times.append(peak_times_row[max_zscore_idx])
        max_zscores.append(max_zscores_row[max_zscore_idx])
        peak_probes.append(peak_probes_row[max_zscore_idx])
    df[overlap_col_name] = overlap_probes
    df["events_row_index"] = overlap_probes_row
    df["global_peak_time"] = peak_times
    df["global_max_zscore"] = max_zscores
    df["peak_probe"] = peak_probes

    # New code to merge overlapping events
    df = df.sort_values("start_time")
    df["group"] = (df["start_time"] > df["end_time"].shift() + offset).cumsum()
    df = (
        df.groupby("group")
        .agg(
            {
                "start_time": "first",
                "end_time": "last",
                overlap_col_name: "first",
                "events_row_index": "first",
                "global_peak_time": "first",
                "global_max_zscore": "first",
                "peak_probe": "first",
            }
        )
        .reset_index(drop=True)
    )

    return df


def find_probe_filename(unfiltered_swr_path, criteria1, criteria2):
    """
    Finds the channel number in the filename of the unfiltered swr file
    Parameters
    ----------
    unfilterd_swr_path : str
        Path to the unfiltered swr file (e.g. '/space/scratch/allen_visbehave_swr_data_test/filtered_lfp/filtered_lfp_0.mat')
    criteria1 : str
        Criteria to find in the filename (e.g. 'filtered_lfp')
    criteria2 : str
        Criteria to find in the filename (e.g. 'channel')
    Returns
    -------
    int : channel number
    """

    filenames = np.array(os.listdir(unfiltered_swr_path))
    filenames2 = np.array(
        [
            file
            for file in os.listdir(unfiltered_swr_path)
            if os.path.isfile(os.path.join(unfiltered_swr_path, file))
        ]
    )
    mask = np.array([], dtype=bool)
    for filename in filenames:
        if (criteria1 in filename) and (criteria2 in filename):
            mask = np.append(mask, True)
        else:
            mask = np.append(mask, False)
    filename = filenames[mask]
    return str(filename[0])


eventspersession_df = pd.read_csv(
    os.path.join(input_dir_global, "eventspersession_df.csv"), index_col=0
)
# Convert 'probe_id' and 'session_id' to a category type

# computing global ripples
session_list = list(eventspersession_df["session_id"].unique())

global_ripples_dict = {}

insufficient_ripples = []

for session_id in session_list:
    try:
        # session_id = session_list[sesh_num]
        sesh_path = os.path.join(input_dir_global, "swrs_session_{}".format(session_id))
        probe_list = eventspersession_df.probe_id[
            eventspersession_df.session_id == session_id
        ].unique()

        probe_event_dict = {}
        probe_list = eventspersession_df.probe_id[
            eventspersession_df.session_id == session_id
        ].unique()

        # check that there are sufficient ripples on at least one probe to pass the criteria
        is_there_enough_ripples = False
        for probe_id in probe_list:

            eventfilename = find_probe_filename(
                sesh_path,
                criteria1="probe_{}".format(probe_id),
                criteria2="karlsson_detector",
            )
            probe_file_path = os.path.join(sesh_path, eventfilename)
            rip_df = pd.read_csv(probe_file_path, index_col=0, compression="gzip")
            if rip_df.shape[0] > minimum_ripple_num:
                is_there_enough_ripples = True
            if no_gamma & (
                rip_df[
                    (rip_df["Overlaps_with_gamma"] == False)
                    & (rip_df["Overlaps_with_movement"] == False)
                ].shape[0]
                > minimum_ripple_num
            ):
                is_there_enough_ripples = True
            elif (
                rip_df[rip_df["Overlaps_with_movement"] == False].shape[0]
                > minimum_ripple_num
            ):
                is_there_enough_ripples = True
        # now we check and skip this iteration if none of the probes have enough
        if is_there_enough_ripples == False:
            insufficient_ripples.append(session_id)
            print(f"{session_id} Does not have enough ripples to proceded")
            continue

        new_probe_list = []
        for probe_id in probe_list:

            eventfilename = find_probe_filename(
                sesh_path,
                criteria1="probe_{}".format(probe_id),
                criteria2="karlsson_detector",
            )
            probe_file_path = os.path.join(sesh_path, eventfilename)
            rips_on_probe = pd.read_csv(
                probe_file_path, index_col=0, compression="gzip"
            )
            if rips_on_probe.shape[0] > minimum_ripple_num:
                probe_event_dict[probe_id] = rips_on_probe
                new_probe_list.append(probe_id)  # add probe_id to new list
            else:
                continue
            # filter out events with movement artifacts, or gamma band events
            filtered_events = probe_event_dict[probe_id]

            if no_gamma:
                filtered_events = filtered_events[
                    (filtered_events.Overlaps_with_gamma == False)
                    & (filtered_events.Overlaps_with_movement == False)
                ]
            else:
                filtered_events = filtered_events[
                    filtered_events.Overlaps_with_movement == False
                ]
            probe_event_dict[probe_id] = filtered_events

        probe_list = new_probe_list  # remove the probes that are not worth analyzing

        # generate times of possible global ripples
        putative_global_ripples = dataframe_union_of_event_times(
            list(probe_event_dict.values()),
            offset=0.02,
            event_name="putative_global_event_id",
        )

        # check each probe for global ripples, give each ripple a global_event_id
        for probe_id in probe_list:

            mask = check_overlap(
                putative_global_ripples, probe_event_dict[probe_id], offset=0.02
            )
            probe_event_df = probe_event_dict[probe_id]

            probe_event_df["putative_global_event_id"] = (
                putative_global_ripples.putative_global_event_id[mask]
            )
            probe_event_dict[probe_id] = probe_event_df
            # at some point in the future we will tag each filtered event with a golbal_ripple id but for now we will not do this
            # probe_event_df.to_csv(os.path.join(sesh_path, 'session_{}_probe_{}_filtered_swrs.csv'.format(session_id, probe_id)), index=True)
        putative_global_ripples = add_overlap_probes(
            putative_global_ripples,
            probe_event_dict,
            "probes_event_is_on",
            offset=merge_events_offset,
        )
        global_ripples_dict[session_id] = putative_global_ripples
        print(global_ripples_dict.keys())
        putative_global_ripples.to_csv(
            os.path.join(
                sesh_path,
                "session_{}_putative_global_swrs_{}.csv".format(
                    session_id, global_rip_label
                ),
            ),
            index=False,
            compression="gzip",
        )
    except:
        traceback.print_exc()
print("Sessions with insufficient ripples...")
print(insufficient_ripples)
