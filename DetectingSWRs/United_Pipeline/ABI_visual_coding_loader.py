# abi_visual_coding_loader.py
import os
import numpy as np
import yaml
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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
            config_content = f.read()
            full_config = yaml.safe_load(config_content)
        dataset_config = full_config["abi_visual_coding"]
        sdk_cache_dir = dataset_config["sdk_cache_dir"]
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
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
        
        # Check for NaNs
        if np.isnan(lfp_ca1).any():
            print(f"NaN detected in LFP data for probe {probe_id}, skipping")
            return None
        
        # Resample to 1500 Hz - using the base class method
        lfp_ca1, lfp_time_index = super().resample_signal(
            lfp_ca1, og_lfp_obj_time_vals, 1500.0
        )
        
        # Find channel with highest ripple power if function provided
        if filter_ripple_band_func is not None:
            lfp_ca1_rippleband = filter_ripple_band_func(lfp_ca1)
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_rippleband)) ** 2
            highest_rip_power = highest_rip_power.max(axis=0)
            
            # Get channel with highest ripple power
            peak_chan_idx = highest_rip_power.argmax()
            this_chan_id = int(lfp_ca1_chans[peak_chan_idx])
            peakrippleband = lfp_ca1_rippleband[:, peak_chan_idx]
            peakripchan_lfp_ca1 = lfp_ca1[:, lfp_ca1_chans == this_chan_id]
            
            # Get channel positions for CA1 channels
            ca1_channel_positions = self.session.channels.loc[lfp_ca1_chans, 'probe_vertical_position']
            
            # Add sharpwave channel detection - using base class implementation
            if hasattr(self, 'sw_component_filter_path') and self.sw_component_filter_path is not None:
                best_sw_chan_id, best_sw_chan_lfp = super().select_sharpwave_channel(
                    ca1_lfp=lfp_ca1,
                    lfp_time_index=lfp_time_index,  
                    ca1_chan_ids=lfp_ca1_chans,
                    this_chan_id=this_chan_id,
                    channel_positions=ca1_channel_positions,
                    ripple_filtered=peakrippleband,
                    filter_path=self.sw_component_filter_path
                )
                best_sw_power_z = zscore(best_sw_chan_lfp)
            else:
                best_sw_chan_id = None
                best_sw_chan_lfp = None
                best_sw_power_z = None
        else:
            peak_chan_idx = None
            this_chan_id = None
            peakrippleband = None
            peakripchan_lfp_ca1 = None
            best_sw_chan_id = None
            best_sw_chan_lfp = None
            best_sw_power_z = None
        del lfp_ca1

        # Collect results
        results = {
            'probe_id': probe_id,
            'lfp_time_index': lfp_time_index,
            'ca1_chans': lfp_ca1_chans,
            'control_lfps': control_channels,
            'control_channels': take_two,
            'peak_ripple_chan_idx': peak_chan_idx,
            'peak_ripple_chan_id': this_chan_id,
            'peak_ripple_chan_raw_lfp': peakripchan_lfp_ca1,
            'chan_id_string': str(this_chan_id) if this_chan_id is not None else None,
            'rippleband': peakrippleband,
            'sharpwave_chan_id': best_sw_chan_id,
            'sharpwave_chan_raw_lfp': best_sw_chan_lfp,
            'sharpwave_power_z': best_sw_power_z
        }
        
        # Return standardized results
        return super().standardize_results(results, 'abi_visual_coding')
    
    def global_events_probe_info(self):
        """
        Get probe-level information needed for global SWR detection.
        
        Returns
        -------
        dict
            Dictionary mapping probe IDs to probe information dictionaries
        """
        
        probe_info = {}
        
        for probe_id in self.probe_id_list:
            # Get units for this probe
            units = self.session.units[self.session.units.probe_id == probe_id]
            
            # Filter for good units based on quality metrics
            good_units = units[
                (units.isolation_distance >= 20) & 
                (units.presence_ratio >= 0.9) &
                (units.isi_violations < 0.5)
            ]
            
            # Get all channels for this probe
            probe_channels = self.session.channels[self.session.channels.probe_id == probe_id]
            
            # Filter for CA1 channels
            ca1_channels = probe_channels[probe_channels.ecephys_structure_acronym == "CA1"]
            
            # Count good units in CA1 by checking if their peak channel is in CA1
            ca1_good_units = 0
            for _, unit in good_units.iterrows():
                if unit.peak_channel_id in ca1_channels.index:
                    ca1_good_units += 1
            
            probe_info[probe_id] = {
                'good_unit_count': len(good_units),
                'ca1_good_unit_count': ca1_good_units,
            }
        
        return probe_info
    def cleanup(self):
        """Cleans up resources to free memory."""
        self.session = None