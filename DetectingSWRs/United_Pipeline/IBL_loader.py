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
        """Get list of probes with CA1 channels."""
        self.get_probe_ids_and_names()
        self.probes_of_interest = []

        for i, probe_id in enumerate(self.probelist):
            channels, _,_ = self.load_channels(i)
            
            if self.has_ca1_channels(channels):
                self.probes_of_interest.append(i)  # Store probe index
        
        print(f"Found {len(self.probes_of_interest)} probes with CA1 channels")
        return self.probes_of_interest
    
    def load_channels(self, probe_idx):
        """Loads channel data for a specific probe."""
        if self.probelist is None:
            self.get_probe_ids_and_names()
        print(f"Loading channel data for probe index: {probe_idx}")
        probe_name = self.probenames[probe_idx]
        probe_id = self.probelist[probe_idx]
        
        # Get channels data
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
        
        # Load bin file
        bin_file = self.load_bin_file(probe_name)
        if bin_file is None:
            return None

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
        #destriped = destripe_lfp(raw, fs=fs_from_sr)
        #print(f"Destriped shape: {destriped.shape}")
        
        fname = f"destriped_probe_{probe_id}.npz"
        path = os.path.join(os.getcwd(), fname)

        # save compressed
        #np.savez_compressed(path, destriped=destriped)

        # later, to load
        destriped_data = np.load(path)
        destriped = destriped_data["destriped"]
        del destriped_data
        
        # normal code...
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
        data = self._load_probe_data(probe_idx) # loads control and destriped data
        if data is None:
            return None
        
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
        
        #Find channel with highest ripple power if function provided
        if filter_ripple_band_func is not None:
            peak_idx, peak_id, peak_lfp, peakrippleband = self.find_peak_ripple_channel(
                lfp_ca1, data['ca1_chans'], filter_ripple_band_func
            )
            
            # Create a channel ID string for naming files
            this_chan_id = f"channelsrawInd_{peak_id}"
            
            # Add sharpwave channel detection if filter path is available
            if hasattr(self, 'sw_component_filter_path') and self.sw_component_filter_path is not None:
                # Get positions of CA1 channels
                try:
                    # Channel positions might be stored differently in IBL data
                    # Try to extract a pandas Series mapping channel IDs to depths
                    ca1_channel_positions = pd.Series(data['ca1_chans'] * 10, index=data['ca1_chans'])
                    
                    best_sw_chan_id, best_sw_chan_lfp = super().select_sharpwave_channel(
                        ca1_lfp=lfp_ca1,
                        lfp_time_index=lfp_time_index,  
                        ca1_chan_ids=data['ca1_chans'],
                        this_chan_id=peak_id,
                        channel_positions=ca1_channel_positions,
                        ripple_filtered=peakrippleband,
                        filter_path=self.sw_component_filter_path
                    )
                    best_sw_chan_lfp = best_sw_chan_lfp.flatten() # debugging
                    best_sw_power_z = zscore(best_sw_chan_lfp)
                except Exception as e:
                    print(f"Warning: Could not select sharpwave channel: {e}")
                    best_sw_chan_id = None
                    best_sw_chan_lfp = None
                    best_sw_power_z = None
            else:
                best_sw_chan_id = None
                best_sw_chan_lfp = None
                best_sw_power_z = None
        else:
            peak_idx = None
            peak_id = None
            peak_lfp = None
            this_chan_id = None
            peakrippleband = None
            best_sw_chan_id = None
            best_sw_chan_lfp = None
            best_sw_power_z = None
        del lfp_ca1
        
        # Collect results
        results = {
            'probe_id': data['probe_id'],
            'probe_name': data['probe_name'],
            'lfp_time_index': lfp_time_index,
            'ca1_chans': data['ca1_chans'],
            'control_lfps': outof_hp_chans_lfp,
            'control_channels': data['control_channels'],
            'peak_ripple_chan_idx': peak_idx,
            'peak_ripple_chan_id': peak_id,
            'peak_ripple_chan_raw_lfp': peak_lfp,
            'chan_id_string': this_chan_id,
            'rippleband': peakrippleband,
            'sharpwave_chan_id': best_sw_chan_id,
            'sharpwave_chan_raw_lfp': best_sw_chan_lfp,
            'sharpwave_power_z': best_sw_power_z,
            'channels': data['channels']
        }
        
        # Return standardized results
        return super().standardize_results(results, 'ibl')
    
    # The remaining methods are IBL-specific and will be directly used by process_probe
    def load_bin_file(self, probe_name):
        """Loads the binary file for a probe."""
        dsets = self.one.list_datasets(
            self.session_id, collection=f"raw_ephys_data/{probe_name}", filename="*.lf.*"
        )
        print(f"Found {len(dsets)} datasets")
        
        self.data_files, _ = self.one.load_datasets(self.session_id, dsets, download_only=False)
        bin_file = next((df for df in self.data_files if df.suffix == ".cbin"), None)
        
        if bin_file is None:
            print(f"No .cbin file found for probe {probe_name}, skipping...")
            
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
    
    def find_peak_ripple_channel(self, lfp_ca1, ca1_chans, filter_ripple_band_func):
        """Finds the CA1 channel with the highest ripple power."""
        lfp_ca1_rippleband = filter_ripple_band_func(lfp_ca1)
        highest_rip_power = np.abs(signal.hilbert(lfp_ca1_rippleband)) ** 2
        highest_rip_power = highest_rip_power.max(axis=0)
        
        peak_channel_idx = highest_rip_power.argmax()
        peak_channel_id = ca1_chans[peak_channel_idx]
        peak_channel_raw_lfp = lfp_ca1[:, peak_channel_idx]
        peakrippleband = lfp_ca1_rippleband[:, peak_channel_idx]
        print(f"Finding channel with highest ripple power...")
        return peak_channel_idx, peak_channel_id, peak_channel_raw_lfp, peakrippleband

    def global_events_probe_info(self):
        """
        Get probe-level information needed for global SWR detection.
        
        Returns
        -------
        dict
            Dictionary mapping probe IDs to probe information dictionaries
        """
        probe_info = {}
        
        for i, probe_id in enumerate(self.probelist):
            # Get the probe name at this index
            probe_name = self.probenames[i]
            
            # Load spike sorting data
            sl = SpikeSortingLoader(eid=self.session_id, pname=probe_name, one=self.one)
            spikes, clusters, channels = sl.load_spike_sorting()
            
            # Merge clusters
            clusters = sl.merge_clusters(spikes, clusters, channels)

            # Filter for good units

            good_units_acronym = clusters.acronym[clusters.label==1.0]
            
            # Count good units in CA1
            ca1_good_units = sum( good_units_acronym == 'CA1')

            probe_info[probe_id] = {
                'good_unit_count': len(good_units_acronym),
                'ca1_good_unit_count': ca1_good_units,
            }
        
        return probe_info

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