# ibl_loader.py
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
from one.api import ONE
import spikeglx
from brainbox.io.one import SpikeSortingLoader, load_channel_locations
from ibldsp.voltage import destripe_lfp
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions
import logging
from math import ceil
from scipy.signal import hilbert
from scipy.stats import skew as stats_skew
import gzip
import json

# Import the BaseLoader
from swr_neuropixels_collection_core import BaseLoader

class ibl_loader(BaseLoader):
    def __init__(self, session_id):
        """Initialize the IBL loader with a session ID."""
        super().__init__(session_id)
        self.probe_id = "Not Loaded Yet"
        self.one_exists = False
        self.probelist = None
        self.probenames = None
        self.one = None
        self.data_files = None
        self.br = None
        
    def set_up(self):
        """Sets up the ONE API connection and brain atlas."""
        # Setup ONE API
        ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
        self.one = ONE(password='international', silent=True)
        self.one_exists = True
        
        # Setup Allen atlas and brain regions if not already provided
        if self.br is None:
            self.ba = AllenAtlas()
            self.br = BrainRegions()
        
        # Load config if available for sharpwave filter path
        try:
            import yaml
            config_path = os.environ.get('CONFIG_PATH', 'unified_detector_config.yaml')
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f.read())
                if 'filters' in full_config and 'sw_component_filter' in full_config['filters']:
                    self.sw_component_filter_path = full_config['filters']['sw_component_filter']
        except Exception as e:
            print(f"Warning: Could not load config for sharp wave filter: {e}")
            self.sw_component_filter_path = None
            
        return self
        
    def get_probe_ids_and_names(self):
        """Gets the probe IDs and names for the session."""
        if not self.one_exists:
            self.set_up()
        probelist, probenames = self.one.eid2pid(self.session_id)
        self.probelist = [str(probe_uuid) for probe_uuid in probelist]
        self.probenames = probenames
        print(f"Probe IDs: {self.probelist}, Probe names: {self.probenames}")
        return self.probelist, self.probenames
    
    def get_probes_with_ca1(self):
        """Get list of probes with CA1 channels. After filtering, set self.probenames and self.probelist to the filtered lists."""
        self.get_probe_ids_and_names()
        filtered_probe_indices = []
        filtered_probe_ids = []
        filtered_probe_names = []

        for i, probe_id in enumerate(self.probelist):
            channels, _, _ = self.load_channels(i)
            if self.has_ca1_channels(channels):
                filtered_probe_indices.append(i)
                filtered_probe_ids.append(self.probelist[i])
                filtered_probe_names.append(self.probenames[i])

        print(f"Found {len(filtered_probe_ids)} probes with CA1 channels")
        # Overwrite self.probenames and self.probelist with filtered lists
        self.probenames = filtered_probe_names
        self.probelist = filtered_probe_ids
        return self.probelist, self.probenames
    
    def load_channels(self, probe_idx):
        """Loads channel data for a specific probe. Now expects probe_idx to be an index into the filtered lists."""
        if self.probelist is None:
            self.get_probe_ids_and_names()
        print(f"Loading channel data for probe index: {probe_idx}")
        if probe_idx >= len(self.probenames):
            raise ValueError(f"Invalid probe index {probe_idx}. Only {len(self.probenames)} probes available.")
        probe_name = self.probenames[probe_idx]
        probe_id = self.probelist[probe_idx]
        collectionname = f"alf/{probe_name}/pykilosort"
        channels = self.one.load_object(self.session_id, "channels", collection=collectionname)
        return channels, probe_name, probe_id
    
    def has_ca1_channels(self, channels):
        """Checks if the channels include CA1."""
        if self.br is None:
            raise ValueError("BrainRegions object (br) must be set to check for CA1 channels")
            
        channels.allen2017_25um_acronym = self.br.id2acronym(
            channels["brainLocationIds_ccf_2017"]
        )
        
        regions_on_probe = np.unique(channels.allen2017_25um_acronym)
        has_ca1 = "CA1" in regions_on_probe
        
        if not has_ca1:
            print(f"No CA1 channels on probe, skipping...")
            
        return has_ca1
    
    def _load_probe_data(self, probe_idx):
        """Load all necessary data for a probe from API.
        
        This method encapsulates all API-specific data loading to separate
        it from processing logic.
        """
        # Load channels
        channels, probe_name, probe_id = self.load_channels(probe_idx)
        
        # Load bin file - load_bin_file now raises FileNotFoundError if missing
        bin_file = self.load_bin_file(probe_name)

        # Read the data
        print(f"Reading LFP data for probe {probe_id}...")
        sr = spikeglx.Reader(bin_file)
        
        # Create time index
        lfp_time_index_og = self.create_time_index(sr, probe_id)
        
        # Extract raw data
        raw, fs_from_sr = self.extract_raw_data(sr)
        del sr  # Free memory
        
        # Destripe data
        print(f"Destripping LFP data for probe {probe_id}...")
        destriped = destripe_lfp(raw, fs=fs_from_sr)
        print(f"Destriped shape: {destriped.shape}")
        del raw  # Free memory
        
        # Get CA1 channels
        lfp_ca1, ca1_chans = self.get_ca1_channels(channels, destriped)
        
        # Get non-hippocampal control channels for artifact detection
        print(f"Getting control channels for artifact detection probe {probe_id}...")
        control_data, control_channels = self.get_non_hippocampal_channels(channels, destriped)
        
        return {
            'probe_id': probe_id,
            'probe_name': probe_name,
            'channels': channels,
            'lfp_ca1': lfp_ca1,
            'ca1_chans': ca1_chans,
            'control_data': control_data,
            'control_channels': control_channels,
            'lfp_time_index_og': lfp_time_index_og,
            'destriped': destriped
        }
    
    def process_probe(self, probe_idx, filter_ripple_band_func=None):
        """Processes a single probe completely."""
        # Separate API access from processing logic
        # _load_probe_data will now raise errors if essential data (like .cbin) is missing
        data = self._load_probe_data(probe_idx) 
        
        # Extract channel positions into a Series.
        # Use 'rawInd' * 10 as a proxy for relative vertical position 
        # This assumes an approximate 10µm spacing related to the raw index.
        # Let potential KeyErrors propagate if 'rawInd' or 'id' are missing.
        # all_channel_positions = pd.Series(data['channels']['axial_um'], index=data['channels']['rawInd'])
        #all_channel_positions = pd.Series(data['channels']['rawInd'] * 10, index=data['channels']['rawInd'])
        axial_um = [ceil(ind/2)*20 for ind in data['channels']['rawInd']]
        all_channel_positions = pd.Series(axial_um, index=data['channels']['rawInd'])
        
        
        #Resample control channels
        outof_hp_chans_lfp = []
        for channel_data in data['control_data']:
            # Reshape to 2D array with shape (1, n_samples)
            channel_data = channel_data.flatten()
            
            # Resample - using base class method
            lfp_control, _ = super().resample_signal(channel_data, data['lfp_time_index_og'], 1500.0)
            
            # Append to list, ensuring correct shape
            outof_hp_chans_lfp.append(lfp_control[:, None])
            del lfp_control  # Free memory
        del data['control_data']  # Free memory
        
        # Resample CA1 channels to 1500 Hz
        print(f"Resampling CA1 channels to 1.5kHz probe {data['probe_id']}...")
        lfp_ca1, lfp_time_index = super().resample_signal(data['lfp_ca1'], data['lfp_time_index_og'], 1500.0)
        del data['destriped']  # Free memory for large array
        
        # Check for NaNs - Log and return None to skip probe
        if np.isnan(lfp_ca1).any():
            # Log error (ensure logger is accessible or use print as fallback)
            # ADD PROBE HAS NaNs to metadata table and skip probe
            logging.warning(f"Session {self.session_id} Probe {data['probe_id']}: NaN detected in resampled CA1 LFP data. Skipping probe.")
            return None # Signal to process_session to skip this probe

        # --- Select Ripple Channel (Now runs unconditionally) --- 
        peak_id, peakrippleband, peak_lfp = self.select_ripple_channel(
            ca1_lfp=lfp_ca1,
            ca1_chan_ids=data['ca1_chans'],
            channel_positions=all_channel_positions, # Pass the extracted positions
            ripple_filter_func=filter_ripple_band_func,
            config=self.config # Pass config from the loader
        )
        
        # Create a channel ID string for naming files
        # This will fail explicitly if peak_id is None (e.g., if select_ripple_channel failed)
        this_chan_id = f"channelsrawInd_{peak_id}"
        
        # --- Select Sharp Wave Channel (Now runs unconditionally) --- 
        best_sw_chan_id, best_sw_chan_lfp = super().select_sharpwave_channel(
            ca1_lfp=lfp_ca1,
            lfp_time_index=lfp_time_index,  
            ca1_chan_ids=data['ca1_chans'],
            peak_ripple_chan_id=peak_id, # Pass selected ripple channel ID
            channel_positions=all_channel_positions,
            ripple_filtered=peakrippleband, # Pass ripple LFP from selected chan
            config=self.config, # Pass config from the loader
            filter_path=getattr(self, 'sw_component_filter_path', None) # Use attribute if exists
        )

        # Extract sharpwave channel information - Remove None check
        # Assumes select_sharpwave_channel raises error or returns valid LFP.
        # If it returns None unexpectedly, the next lines will raise an explicit error.
        best_sw_chan_lfp = best_sw_chan_lfp.flatten()
        best_sw_power_z = zscore(best_sw_chan_lfp) # Calculate Z-score if needed elsewhere

        del lfp_ca1
        
        # Collect results using final, consistent key names
        results = {
            'probe_id': data['probe_id'],
            'probe_name': data['probe_name'], # IBL specific?
            'lfp_time_index': lfp_time_index,
            'sampling_rate': 1500.0, # Explicitly add sampling rate
            'dataset_type': 'ibl', # Explicitly add dataset type
            'ca1_channel_ids': data['ca1_chans'], # Renamed
            'control_lfps': outof_hp_chans_lfp, # This contains the LFP data for control channels
            'control_channel_ids': data['control_channels'], # Renamed
            'peak_ripple_chan_id': peak_id, # ID of the selected ripple channel
            'peak_ripple_raw_lfp': peak_lfp, # Renamed - Raw LFP from selected ripple channel
            # 'chan_id_string': this_chan_id, # Removed - Redundant info?
            'ripple_band_filtered': peakrippleband, # Renamed - Filtered LFP from selected ripple channel
            'sharpwave_chan_id': best_sw_chan_id, # ID of the selected SW channel
            'sharpwave_chan_raw_lfp': best_sw_chan_lfp, # Raw LFP from selected SW channel
            'sharpwave_power_z': best_sw_power_z, # Z-scored power (or None)
            'channels': data['channels'], # IBL specific? Full channel info table?
            'channel_selection_metadata': self.channel_selection_metadata_dict # Metadata dict added earlier
        }
        
        # Return results directly, without standardization call
        return results
    
    # The remaining methods are IBL-specific and will be directly used by process_probe
    def load_bin_file(self, probe_name):
        """Loads the binary file for a probe."""
        
        dsets = self.one.list_datasets(
            self.session_id, collection=f"raw_ephys_data/{probe_name}", filename="*.lf.*"
        )
        print(f"Found {len(dsets)} datasets")
        
        if not dsets:
            raise FileNotFoundError(f"No datasets found for session {self.session_id}, probe {probe_name}")
        
        self.data_files, _ = self.one.load_datasets(self.session_id, dsets, download_only=False)
        
        # Check if data_files is None or empty
        if not self.data_files:
            raise FileNotFoundError(f"Failed to load datasets for session {self.session_id}, probe {probe_name}")
        
        bin_file = next((df for df in self.data_files if df.suffix == ".cbin"), None)
        
        if bin_file is None:
            # Raise FileNotFoundError if the essential .cbin file is missing
            raise FileNotFoundError(f"No .cbin file found for session {self.session_id}, probe {probe_name} in datasets: {dsets}")
            
        return bin_file
    
    def create_time_index(self, sr, probe_id):
        """Creates a time index for the LFP data."""
        ssl = SpikeSortingLoader(pid=probe_id, one=self.one)
        t0 = ssl.samples2times(0, direction="forward")
        dt = (ssl.samples2times(1, direction="forward") - t0) * 12
        lfp_time_index_og = np.arange(0, sr.shape[0]) * dt + t0
        del ssl
        
        return lfp_time_index_og
    
    def extract_raw_data(self, sr):
        """Extracts raw data from the SpikeGLX reader."""
        raw = sr[:, : -sr.nsync].T
        fs_from_sr = sr.fs
        
        return raw, fs_from_sr

    def get_ca1_channels(self, channels, destriped):
        """Gets the CA1 channels from the destriped data."""
        # add acronyms to data
        channels.allen2017_25um_acronym = self.br.id2acronym(
            channels["brainLocationIds_ccf_2017"]
        )
        # filter for ca1 channels
        ca1_chans = channels.rawInd[channels.allen2017_25um_acronym == "CA1"]
        lfp_ca1 = destriped[ca1_chans, :]
        
        return lfp_ca1, ca1_chans
    
    def get_non_hippocampal_channels(self, channels, destriped):
        """Gets two non-hippocampal channels for artifact detection."""
        # Find channels outside the hippocampal formation
        not_a_hp_chan = np.logical_not(
            np.isin(
                channels.allen2017_25um_acronym,
                ["CA3", "CA2", "CA1", "HPF", "EC", "DG"],
            )
        )
        
        # Select two random non-hippocampal channels
        control_channels = np.random.choice(
            channels.rawInd[not_a_hp_chan], 2, replace=False
        )
        
        # Extract data for these channels
        control_data = []
        for channel_idx in control_channels:
            control_data.append(destriped[channel_idx, :].flatten())
            
        return control_data, control_channels
    

    def cleanup(self):
        """Cleans up resources to free memory."""
        del self.one
        if hasattr(self, 'data_files') and self.data_files:
            try:
                s = str(self.data_files[0])
                index = s.find("raw_ephys_data")
                s = s[:index]
                s = s.replace("PosixPath('", "")
                s = s.rstrip("/")
                cmd = f"rm -rf {s}"
                os.system(cmd)
            except Exception as e:
                print(f"Warning: Could not clean up data files: {e}")

    def get_metadata_for_probe(self, probe_id, config=None):
        """
        Generates metadata for a single specified probe (IBL).
        Note: Requires loading SpikeSorting data, which can be slow.

        Parameters
        ----------
        probe_id : str
            The unique identifier (UUID string) for the probe being processed.
        config : dict, optional
            Configuration dictionary (not used in this implementation).

        Returns
        -------
        dict
            Standardized probe metadata.
        """
        # --- Basic Setup ---
        # Assume ONE API and probe list/names are loaded via set_up / get_probe_ids_and_names
        # Let access fail explicitly if they aren't.
        # Create dict to store results for this specific probe call
        metadata = {
            'probe_id': probe_id, # Included for clarity, though caller has it
            'ca1_channel_count': 0,
            'ca1_span_microns': 0.0,
            'total_unit_count': 0,
            'good_unit_count': 0,
            'ca1_total_unit_count': 0,
            'ca1_good_unit_count': 0
        }
        # --- Find probe name corresponding to probe_id (UUID) ---
        # Use filtered lists
        if probe_id not in self.probelist:
            raise ValueError(f"Probe ID {probe_id} not found in filtered CA1 probe list.")
        probe_idx = self.probelist.index(probe_id)
        probe_name = self.probenames[probe_idx]
        # --- Load Spike Sorting Data ---
        print(f"Loading spike sorting data for probe {probe_name} ({probe_id})...")
        sl = SpikeSortingLoader(eid=self.session_id, pname=probe_name, one=self.one, atlas=self.ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.metrics'])
        clusters = sl.merge_clusters(spikes, clusters, channels)
        clusters = clusters.to_df()
        print(f"Spike sorting data loaded for probe {probe_id}")
        # --- Calculate Metadata (Allow KeyErrors/AttributeErrors to propagate) ---
        metadata['total_unit_count'] = len(clusters)
        good_units_mask = (clusters.label == 1)
        metadata['good_unit_count'] = np.sum(good_units_mask)
        ca1_good_units = np.sum((clusters.acronym == 'CA1') & good_units_mask)
        metadata['ca1_good_unit_count'] = ca1_good_units
        ca1_mask = (channels.acronym == 'CA1')
        metadata['ca1_channel_count'] = np.sum(ca1_mask)
        ca1_depths = channels.axial_um[ca1_mask]
        metadata['ca1_span_microns'] = float(ca1_depths.max() - ca1_depths.min())
        ca1_channel_ids = channels.rawInd[ca1_mask]
        in_ca1_mask = np.isin(clusters.channels, ca1_channel_ids)
        metadata['ca1_total_unit_count'] = np.sum(in_ca1_mask)
        del sl, spikes, clusters, channels
        return metadata