#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from one.api import ONE
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
from IBL_loader import ibl_loader
import spikeglx

# Session ID that's causing the error
SESSION_ID = "3555ce3a-ccce-4b7e-8eb1-c2571913599e"

# Initialize the loader
loader = ibl_loader(SESSION_ID)
loader.set_up()

# Get probe IDs and names
print("Getting probe IDs and names...")
probelist, probenames = loader.get_probe_ids_and_names()
print(f"Found {len(probelist)} probes: {probenames}")

# For each probe, let's inspect the channel data
i = 0  # Start with first probe
probe_id = probelist[i]
print(f"\nProcessing probe {i+1}/{len(probelist)}: {probe_id}")

# Load channels
channels, probe_name, probe_id = loader.load_channels(i)

# Print channel data info
print("\nChannel data info:")
print(f"Channel data type: {type(channels)}")
print(f"Channel data columns: {channels.columns.tolist()}")
print("\nFirst few rows of channel data:")
print(channels.head())

# Load bin file and get raw data
bin_file = loader.load_bin_file(probe_name)
sr = spikeglx.Reader(bin_file)
raw, fs_from_sr = loader.extract_raw_data(sr)
del sr  # Free memory

# Destripe data
print(f"Destriping LFP data for probe {probe_id}...")
destriped = destripe_lfp(raw, fs=fs_from_sr)
print(f"Destriped shape: {destriped.shape}")
del raw  # Free memory

# Save destriped data for later use
# np.save(f'destriped_data_{SESSION_ID}_{probe_id}.npy', destriped)

# Load previously saved destriped data
# destriped = np.load(f'destriped_data_{SESSION_ID}_{probe_id}.npy')

# Get CA1 channels
lfp_ca1, ca1_chans = loader.get_ca1_channels(channels, destriped)

# Get non-hippocampal control channels
control_data, control_channels = loader.get_non_hippocampal_channels(channels, destriped)

# Create time index
lfp_time_index_og = loader.create_time_index(sr, probe_id)
# Now you can inspect and debug the data structures
# For example:
print("\nCA1 channel info:")
print(f"Number of CA1 channels: {len(ca1_chans)}")
print(f"CA1 channel IDs: {ca1_chans}")

print("\nControl channel info:")
print(f"Number of control channels: {len(control_channels)}")
print(f"Control channel IDs: {control_channels}")

# You can now interactively explore the data structures
# and debug the has_ca1_channels method

