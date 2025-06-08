# abi_visual_coding_loader.py
import os
import numpy as np
import yaml
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pandas as pd
import logging

# Import the BaseLoader
from swr_neuropixels_collection_core import BaseLoader

class abi_visual_coding_loader(BaseLoader):
    def __init__(self, session_id):
        """Initialize the ABI loader with a session ID."""
        super().__init__(session_id)
        self.cache = None
        self.session = None
        self.probe_id_list = None
        self.probes_of_interest = None
        
    def set_up(self, cache_directory=None):
        """Sets up the EcephysProjectCache and loads the session."""
        # Set up the cache
        config_path = os.environ.get('CONFIG_PATH', 'united_detector_config.yaml')
        with open(config_path, "r") as f:
            # Read the raw YAML content as a string
            raw_yaml_content = f.read()
            # Expand environment variables (handles $VAR or ${VAR})
            expanded_yaml_content = os.path.expandvars(raw_yaml_content)
            # Load the YAML from the expanded string
            full_config = yaml.safe_load(expanded_yaml_content)
            
        dataset_config = full_config["abi_visual_coding"]
        # Prioritize environment variable for cache path, fallback to config
        sdk_cache_dir = os.environ.get('ABI_VISUAL_CODING_SDK_CACHE', dataset_config["sdk_cache_dir"])
        
        # Ensure the cache directory exists before creating the manifest path
        if not os.path.isdir(sdk_cache_dir):
             raise FileNotFoundError(f"Specified SDK cache directory does not exist: {sdk_cache_dir}")
             
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
        
        # Check if manifest exists, create if not (as per AllenSDK example for first run)
        # Although from_warehouse usually expects it, this handles first-time setup better.
        if not os.path.exists(manifest_path):
            print(f"Manifest file not found at {manifest_path}. Attempting to initialize cache...")
            # Ensure parent directory exists before trying to initialize
            os.makedirs(sdk_cache_dir, exist_ok=True) 
            # Initialize cache to create manifest - may download small session/probe files
            try:
                print("Initializing EcephysProjectCache to potentially create manifest...")
                temp_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
                # Attempt a basic operation to trigger manifest creation/download if needed
                _ = temp_cache.get_session_table()
                print("Cache initialized successfully.")
                del temp_cache
            except Exception as e_init:
                print(f"Error initializing cache/creating manifest: {e_init}")
                # Decide how to handle this - raise error or proceed cautiously?
                # For now, let's raise to make the issue explicit
                raise RuntimeError(f"Failed to initialize cache or create manifest at {manifest_path}") from e_init
        
        # Now, load the cache assuming the manifest exists
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        
        # Set the sharp wave component filter path
        if 'filters' in full_config and 'sw_component_filter' in full_config['filters']:
            self.sw_component_filter_path = full_config['filters']['sw_component_filter']

        # Load the session
        self.session = self.cache.get_session_data(self.session_id)
        
        print(f"Session {self.session_id} loaded")
        return self
    
    def has_ca1_channels(self):
        """Checks if the session includes CA1 channels."""
        has_ca1 = np.isin("CA1", list(self.session.channels.ecephys_structure_acronym.unique()))
        
        if not has_ca1:
            print(f"Session {self.session_id} does not have CA1 channels")
            
        return has_ca1
    
    def get_probes_with_ca1(self):
        """Gets the list of probes that have CA1 channels."""
        # Get probes for this session
        self.probe_id_list = [int(item) for item in self.session.channels.probe_id.unique()]
        
        # Find probes with CA1 channels
        self.probes_of_interest = []
        for probe_id in self.probe_id_list:
            has_ca1_and_exists = np.isin(
                "CA1",
                list(
                    self.session.channels[
                        self.session.channels.probe_id == probe_id
                    ].ecephys_structure_acronym.unique()
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
        probe_channels = self.session.channels[self.session.channels.probe_id == probe_id] # Filter channels DataFrame
        # Create Series mapping channel ID to vertical position
        # Check column name for Visual Coding - might be ecephys_probe_vertical_position? Assuming probe_vertical_position for now.
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
                organisedprobechans.ecephys_structure_acronym,
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
            # Resample to match CA1 data - using the base class method
            movement_control_channel, lfp_time_index = super().resample_signal(
                movement_control_channel, lfp.time.values, 1500.0
            )
            movement_control_channel = movement_control_channel[:, None]
            control_channels.append(movement_control_channel)
        
        # Get CA1 channels for this probe
        ca1_chans = self.session.channels.probe_channel_number[
            (self.session.channels.probe_id == probe_id)
            & (self.session.channels.ecephys_structure_acronym == "CA1")
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
        
        # Resample to 1500 Hz - using the base class method
        lfp_ca1, lfp_time_index = super().resample_signal(
            lfp_ca1, og_lfp_obj_time_vals, 1500.0
        )
        
        # Find channel with highest ripple power - REMOVED conditional logic
        # --- Select Ripple Channel (Now runs unconditionally) --- 
        this_chan_id, peakrippleband, peakripchan_lfp_ca1 = self.select_ripple_channel(
            ca1_lfp=lfp_ca1,
            ca1_chan_ids=lfp_ca1_chans,
            channel_positions=all_channel_positions, # Pass the extracted positions
            ripple_filter_func=filter_ripple_band_func, # Assumed to be valid
            config=None # Pass config if needed
        )

        # --- Select Sharp Wave Channel (Now runs unconditionally) --- 
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

        # Extract sharpwave channel information - Remove None check
        # Assumes select_sharpwave_channel raises error or returns valid LFP.
        # If it returns None unexpectedly, the next lines will raise an explicit error.
        best_sw_chan_lfp = best_sw_chan_lfp.flatten()
        best_sw_power_z = zscore(best_sw_chan_lfp)

        del lfp_ca1

        # Collect results using final, consistent key names
        results = {
            'probe_id': probe_id,
            'lfp_time_index': lfp_time_index,
            'sampling_rate': 1500.0, # Explicitly add sampling rate
            'dataset_type': 'abi_visual_coding', # Explicitly add dataset type
            'ca1_channel_ids': lfp_ca1_chans, # Renamed
            'control_lfps': control_channels, # LFP data for control channels
            'control_channel_ids': take_two, # Renamed? (was control_channels_ids)
            'peak_ripple_chan_id': this_chan_id,
            'peak_ripple_raw_lfp': peakripchan_lfp_ca1, # Renamed
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
        Generates metadata for a single specified probe (ABI Visual Coding).

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
        # Assume session, channels, units are loaded via set_up.
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

        # --- Filter Data for Probe ---
        # Allow potential KeyErrors etc. to propagate and cause failure
        probe_channels = self.session.channels[self.session.channels.probe_id == probe_id]
        probe_units = self.session.units[self.session.units.probe_id == probe_id]
            
        # Subsequent operations will fail explicitly if probe_channels is unexpectedly empty

        metadata['total_unit_count'] = len(probe_units)

        # --- Identify Good Units (Allen SDK definition using metrics) ---
        # Allow potential KeyErrors to propagate if columns are missing
        good_units = probe_units[
            (probe_units.isolation_distance >= 20) & 
            (probe_units.presence_ratio >= 0.9) &
            (probe_units.isi_violations < 0.5)
            ]
        metadata['good_unit_count'] = len(good_units)

        # --- CA1 Analysis ---
        # Allow potential KeyErrors to propagate
        ca1_channels = probe_channels[probe_channels.ecephys_structure_acronym == "CA1"]
        metadata['ca1_channel_count'] = len(ca1_channels)

        # Calculate CA1 span (unconditionally, assuming >1 channel)
        ca1_depths = ca1_channels['probe_vertical_position']
        metadata['ca1_span_microns'] = float(ca1_depths.max() - ca1_depths.min())
        
        # Calculate CA1 unit counts (unconditionally)
        ca1_channel_ids = ca1_channels.index
        units_in_ca1 = probe_units[probe_units['peak_channel_id'].isin(ca1_channel_ids)]
        metadata['ca1_total_unit_count'] = len(units_in_ca1)
        
        # Good units in CA1 (using the pre-filtered good_units DataFrame)
        good_units_in_ca1 = good_units[good_units['peak_channel_id'].isin(ca1_channel_ids)]
        metadata['ca1_good_unit_count'] = len(good_units_in_ca1)

        return metadata