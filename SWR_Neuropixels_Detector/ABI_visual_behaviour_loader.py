# abi_visual_behaviour_loader.py
import os
import numpy as np
import yaml
from scipy import signal
from scipy.stats import zscore
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
import pandas as pd
import logging

# Import the BaseLoader
from swr_neuropixels_collection_core import BaseLoader

class abi_visual_behaviour_loader(BaseLoader):
    def __init__(self, session_id):
        """Initialize the ABI loader with a session ID."""
        super().__init__(session_id)
        self.cache = None
        self.session = None
        self.probe_id_list = None
        self.probes_of_interest = None
        self.sw_channel_info = None
        
    def set_up(self, cache_directory=None):
        """Sets up the EcephysProjectCache and loads the session."""
        # Set up the cache
        config_path = os.environ.get('CONFIG_PATH', 'expanded_config.yaml')
        
        with open(config_path, "r") as f:
            raw_content = f.read()
            # Replace environment variables
            for key, value in os.environ.items():
                raw_content = raw_content.replace(f"${key}", value)
            full_config = yaml.safe_load(raw_content)
        dataset_config = full_config["abi_visual_behaviour"]
        sdk_cache_dir = dataset_config["sdk_cache_dir"]
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
        self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=sdk_cache_dir)

        # provide the sharp wave component band filter
        self.sw_component_filter_path = full_config['filters']['sw_component_filter']
        
        # Load the session
        self.session = self.cache.get_ecephys_session(ecephys_session_id=self.session_id)
        self.session.channels = self.session.get_channels()
        
        print(f"Session {self.session_id} loaded")
        return self
    
    def has_ca1_channels(self):
        """Checks if the session includes CA1 channels."""
        has_ca1 = np.isin("CA1", list(self.session.channels.structure_acronym.unique()))
        
        if not has_ca1:
            print(f"Session {self.session_id} does not have CA1 channels")
            
        return has_ca1
    
    def get_probes_with_ca1(self):
        """Gets the list of probes that have CA1 channels."""
        # Get probes with LFP data
        probes_table_df = self.cache.get_probe_table()
        valid_lfp = probes_table_df[probes_table_df["has_lfp_data"]]
        
        # Get probes for this session
        self.probe_id_list = list(
            valid_lfp[valid_lfp.ecephys_session_id == self.session_id].index
        )
        
        # Find probes with CA1 channels
        self.probes_of_interest = []
        for probe_id in self.probe_id_list:
            has_ca1_and_exists = np.isin(
                "CA1",
                list(
                    self.session.channels[
                        self.session.channels.probe_id == probe_id
                    ].structure_acronym.unique()
                ),
            )
            if has_ca1_and_exists:
                self.probes_of_interest.append(probe_id)
        
        print(f"Found {len(self.probes_of_interest)} probes with CA1 channels")
        return self.probes_of_interest

    def process_probe(self, probe_id, filter_ripple_band_func=None):
        """Processes a single probe to extract CA1 and control channels."""
        print(f"Processing probe: {probe_id}")
        
        # Get LFP for the probe
        lfp = self.session.get_lfp(probe_id)
        og_lfp_obj_time_vals = lfp.time.values
        
        # --- Define all_channel_positions --- 
        # Get channel info for this probe
        probe_channels = self.session.channels[self.session.channels.probe_id == probe_id]
        # Create Series mapping channel ID to vertical position
        all_channel_positions = pd.Series(probe_channels['probe_vertical_position'].values, index=probe_channels.index)
        # --- End definition ---

        # Get control channels outside hippocampus
        idx = self.session.channels.probe_id == probe_id
        organisedprobechans = self.session.channels[idx].sort_values(
            by="probe_vertical_position"
        )
        organisedprobechans = organisedprobechans[
            np.isin(organisedprobechans.index.values, lfp.channel.values)
        ]
        
        # Find channels outside hippocampus
        not_a_ca1_chan = np.logical_not(
            np.isin(
                organisedprobechans.structure_acronym,
                ["CA3", "CA2", "CA1", "HPF", "EC", "DG"],
            )
        )
        
        # Choose two random channels
        take_two = np.random.choice(
            organisedprobechans.index[not_a_ca1_chan], 2, replace=False
        )
        control_channels = []
        
        # Get LFP for control channels
        for channel_outside_hp in take_two:
            movement_control_channel = lfp.sel(channel=channel_outside_hp)
            movement_control_channel = movement_control_channel.to_numpy()
            # Resample to match CA1 data
            movement_control_channel, lfp_time_index = super().resample_signal(
                movement_control_channel, lfp.time.values, 1500.0
            )
            movement_control_channel = movement_control_channel[:, None]
            control_channels.append(movement_control_channel)
        
        # Get CA1 channels for this probe
        ca1_chans = self.session.channels.probe_channel_number[
            (self.session.channels.probe_id == probe_id)
            & (self.session.channels.structure_acronym == "CA1")
        ]
        ca1_idx = np.isin(lfp.channel.values, ca1_chans.index.values)
        ca1_idx = lfp.channel.values[ca1_idx]
        
        # Select CA1 channels
        lfp_ca1 = lfp.sel(channel=ca1_idx)
        del lfp
        lfp_ca1 = lfp_ca1.to_pandas()
        lfp_ca1_chans = lfp_ca1.columns
        lfp_ca1 = lfp_ca1.to_numpy()
        
        # Check for NaNs - Log and return None to skip probe
        if np.isnan(lfp_ca1).any():
            # Log error (ensure logger is accessible or use print as fallback)
            # ADD PROBE HAS NaNs to metadata table and skip probe
            logging.warning(f"Session {self.session_id} Probe {probe_id}: NaN detected in resampled CA1 LFP data. Skipping probe.")
            return None # Signal to process_session to skip this probe

        # Resample to 1500 Hz
        lfp_ca1, lfp_time_index = super().resample_signal(
            lfp_ca1, og_lfp_obj_time_vals, 1500.0
        )
        
        # Find channel with highest ripple power if function provided
        # --- Select Ripple Channel --- 
        this_chan_id, peakrippleband, peakripchan_lfp_ca1 = self.select_ripple_channel(
            ca1_lfp=lfp_ca1,
            ca1_chan_ids=lfp_ca1_chans,
            channel_positions=all_channel_positions, # Pass the extracted positions
            ripple_filter_func=filter_ripple_band_func,
            config=None # Pass config if needed
        )


        # --- Select Sharp Wave Channel --- 
        best_sw_chan_id, best_sw_chan_lfp = super().select_sharpwave_channel(
            ca1_lfp=lfp_ca1,
            lfp_time_index=lfp_time_index,  
            ca1_chan_ids=lfp_ca1_chans,
            peak_ripple_chan_id=this_chan_id, # Pass selected ripple channel ID
            channel_positions=all_channel_positions,
            ripple_filtered=peakrippleband,
            config=None, # Pass config if needed
            filter_path=getattr(self, 'sw_component_filter_path', None) # Use attribute if exists
        )

        # Extract sharpwave channel information 
        best_sw_chan_lfp = best_sw_chan_lfp.flatten()
        best_sw_power_z = zscore(best_sw_chan_lfp)

        del lfp_ca1

        # Collect results using final, consistent key names
        results = {
            'probe_id': probe_id,
            'lfp_time_index': lfp_time_index,
            'sampling_rate': 1500.0, # Explicitly add sampling rate
            'dataset_type': 'abi_visual_behaviour', # Explicitly add dataset type
            'ca1_channel_ids': lfp_ca1_chans, # Renamed
            'control_lfps': control_channels, # LFP data for control channels
            'control_channel_ids': take_two, # Renamed? (was control_channels_ids)
            'peak_ripple_chan_id': this_chan_id,
            'peak_ripple_raw_lfp': peakripchan_lfp_ca1, # Renamed
            # 'chan_id_string': str(this_chan_id), # Removed - Redundant
            'ripple_band_filtered': peakrippleband, # Renamed
            'sharpwave_chan_id': best_sw_chan_id,
            'sharpwave_chan_raw_lfp': best_sw_chan_lfp,
            'sharpwave_power_z': best_sw_power_z,
            'channel_selection_metadata': self.channel_selection_metadata_dict
        }
        
        # Return results directly, without standardization call
        return results
    
    def cleanup(self):
        """Cleans up resources to free memory."""
        self.session = None

    def get_metadata_for_probe(self, probe_id, config=None):
        """
        Generates metadata for a single specified probe (ABI Visual Behaviour).

        Parameters
        ----------
        probe_id : int
            The unique identifier for the probe being processed.
        config : dict, optional
            Configuration dictionary (not used in this implementation).

        Returns
        -------
        dict
            Standardized probe metadata.
        """
        # --- Basic Setup ---
        # Assume session, channels, cache are loaded via set_up.
        # Let access fail explicitly (AttributeError) if they aren't.
        metadata = {
            'probe_id': probe_id,
            'ca1_channel_count': 0,
            'ca1_span_microns': 0.0,
            'total_unit_count': 0,
            'good_unit_count': 0,
            'ca1_total_unit_count': 0,
            'ca1_good_unit_count': 0
        }

        # --- Get Data for Probe ---
        # Allow potential KeyErrors etc. to propagate
        probe_channels = self.session.channels[self.session.channels.probe_id == probe_id]
        session_units = self.cache.get_unit_table()
        probe_units = session_units[session_units.ecephys_probe_id == probe_id]

        # Subsequent operations will fail explicitly if probe_channels is unexpectedly empty

        metadata['total_unit_count'] = len(probe_units)

        # --- Identify Good Units (Uses 'quality' column) ---
        # Allow potential KeyError to propagate if 'quality' column is missing
        good_units = probe_units[probe_units.quality == 'good']
        metadata['good_unit_count'] = len(good_units)

        # --- CA1 Analysis ---
        # Allow potential KeyErrors to propagate
        ca1_channels = probe_channels[probe_channels.structure_acronym == "CA1"]
        metadata['ca1_channel_count'] = len(ca1_channels)

        # Calculate CA1 span (unconditionally, assuming >1 channel)
        ca1_depths = ca1_channels['probe_vertical_position']
        metadata['ca1_span_microns'] = float(ca1_depths.max() - ca1_depths.min())

        # Calculate CA1 unit counts (unconditionally)
        ca1_channel_ids = ca1_channels.index
        units_in_ca1 = probe_units[probe_units['ecephys_channel_id'].isin(ca1_channel_ids)]
        metadata['ca1_total_unit_count'] = len(units_in_ca1)

        # Good units in CA1 (using the pre-filtered good_units DataFrame)
        good_units_in_ca1 = good_units[good_units['ecephys_channel_id'].isin(ca1_channel_ids)]
        metadata['ca1_good_unit_count'] = len(good_units_in_ca1)

        return metadata