# IBL Loader
import time
import numpy as np
from scipy import signal, interpolate
from one.api import ONE
import spikeglx
from brainbox.io.one import SpikeSortingLoader, load_channel_locations
from neurodsp.voltage import destripe_lfp
import ibllib.atlas
from iblatlas.atlas import AllenAtlas
from iblatlas.regions import BrainRegions



class ibl_loader:
    def __init__(self, session_id):
        """
        Initialize the IBL loader with a session ID.
        
        Parameters
        ----------
        session_id : str
            The IBL session ID
        br : BrainRegions, optional
            An existing BrainRegions object to use. If None, brain regions
            will need to be added externally.
        """
        self.session_id = session_id
        self.probe_id = "Not Loaded Yet"
        self.one_exists = False
        self.probelist = None
        self.probenames = None
        self.one = None
        self.data_files = None
        self.br = None
        
    def set_up(self):
        """
        Sets up the ONE API connection and brain atlas.
        
        Returns
        -------
        self : ibl_loader
            Returns the instance for method chaining.
        """
        # Setup ONE API
        ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
        self.one = ONE(password='international', silent=True)
        self.one_exists = True
        
        # Setup Allen atlas and brain regions if not already provided
        if self.br is None:
            self.ba = AllenAtlas()
            self.br = BrainRegions()
            
        return self
        
    def get_probe_ids_and_names(self):
        """
        Gets the probe IDs and names for the session.        
        """
        if not self.one_exists:
            self.set_up()
        self.probelist, self.probenames = self.one.eid2pid(self.session_id)
        print(f"Probe IDs: {self.probelist}, Probe names: {self.probenames}")
        return self.probelist, self.probenames
    
    def load_channels(self, probe_idx):
        """
        Loads channel data for a specific probe.
        
        Parameters
        ----------
        probe_idx : int
            Index of the probe in the probelist.
            
        Returns
        -------
        tuple
            (channels, probe_name, probe_id)
        """
        if self.probelist is None:
            self.get_probe_ids_and_names()
            
        probe_name = self.probenames[probe_idx]
        probe_id = self.probelist[probe_idx]
        print(f"Loading channel data for probe: {probe_id}")
        
        # Get channels data
        collectionname = f"alf/{probe_name}/pykilosort"
        channels = self.one.load_object(self.session_id, "channels", collection=collectionname)
        
        return channels, probe_name, probe_id
    
    def has_ca1_channels(self, channels):
        """
        Checks if the channels include CA1.
        
        Parameters
        ----------
        channels : object
            Channel information object
            
        Returns
        -------
        bool
            True if CA1 channels exist, False otherwise
        """
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
    
    def load_bin_file(self, probe_name):
        """
        Loads the binary file for a probe.
        
        Parameters
        ----------
        probe_name : str
            Name of the probe
            
        Returns
        -------
        pathlib.Path or None
            Path to the binary file
        """
        # Find the relevant datasets and download them
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
        """
        Creates a time index for the LFP data.
        
        Parameters
        ----------
        sr : spikeglx.Reader
            SpikeGLX reader object
        probe_id : str
            Probe ID
            
        Returns
        -------
        numpy.ndarray
            Time index for the LFP data
        """
        # Make time index
        start_time = time.time()
        ssl = SpikeSortingLoader(pid=probe_id, one=self.one)
        t0 = ssl.samples2times(0, direction="forward")
        dt = (ssl.samples2times(1, direction="forward") - t0) * 12
        lfp_time_index_og = np.arange(0, sr.shape[0]) * dt + t0
        del ssl
        print(f"Time index created, time elapsed: {time.time() - start_time}")
        
        return lfp_time_index_og
    
    def extract_raw_data(self, sr):
        """
        Extracts raw data from the SpikeGLX reader.
        
        Parameters
        ----------
        sr : spikeglx.Reader
            SpikeGLX reader object
            
        Returns
        -------
        tuple
            (raw_data, sampling_rate)
        """
        # Extract raw data
        start_time = time.time()
        raw = sr[:, : -sr.nsync].T
        fs_from_sr = sr.fs
        print(f"Raw data extracted, time elapsed: {time.time() - start_time}")
        
        return raw, fs_from_sr
    
    def destripe_data(self, raw, fs):
        """
        Applies destriping to the raw data.
        
        Parameters
        ----------
        raw : numpy.ndarray
            Raw data
        fs : float
            Sampling rate
            
        Returns
        -------
        numpy.ndarray
            Destriped data
        """
        start_time = time.time()
        destriped = destripe_lfp(raw, fs=fs)
        print(f"Destriped shape: {destriped.shape}")
        print(f"Destriping done, time elapsed: {time.time() - start_time}")
        
        return destriped
    
    def get_ca1_channels(self, channels, destriped):
        """
        Gets the CA1 channels from the destriped data.
        
        Parameters
        ----------
        channels : object
            Channel information object
        destriped : numpy.ndarray
            Destriped data
            
        Returns
        -------
        tuple
            (ca1_lfp, ca1_channel_indices)
        """
        ca1_chans = channels.rawInd[channels.allen2017_25um_acronym == "CA1"]
        lfp_ca1 = destriped[ca1_chans, :]
        
        return lfp_ca1, ca1_chans
    
    def get_non_hippocampal_channels(self, channels, destriped):
        """
        Gets two non-hippocampal channels for artifact detection.
        
        Parameters
        ----------
        channels : object
            Channel information object
        destriped : numpy.ndarray
            Destriped data
            
        Returns
        -------
        tuple
            (non_hippocampal_lfp_list, non_hippocampal_channel_indices)
        """
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
            control_data.append(destriped[channel_idx, :])
            
        return control_data, control_channels
    
    def resample_signal(self, lfp_data, time_index_og, target_fs=1500.0):
        """
        Resamples the signal to a target frequency.
        
        Parameters
        ----------
        lfp_data : numpy.ndarray
            LFP data
        time_index_og : numpy.ndarray
            Original time index
        target_fs : float, optional
            Target sampling frequency
            
        Returns
        -------
        tuple
            (resampled_data, new_time_index)
        """
        # Create new time index at target sampling rate
        t_start = time_index_og[0]
        t_end = time_index_og[-1]
        dt_new = 1.0 / target_fs
        n_samples = int(np.ceil((t_end - t_start) / dt_new))
        new_time_index = t_start + np.arange(n_samples) * dt_new
        
        # Check if lfp_data is 1D or 2D
        if lfp_data.ndim == 1:
            # For 1D array
            interp_func = interpolate.interp1d(
                time_index_og,
                lfp_data,
                bounds_error=False,
                fill_value="extrapolate",
            )
            resampled = interp_func(new_time_index)
            resampled = resampled.T  # Transpose for standard orientation
        else:
            # For 2D array (multiple channels)
            resampled = np.zeros((lfp_data.shape[0], len(new_time_index)))
            for i in range(lfp_data.shape[0]):
                interp_func = interpolate.interp1d(
                    time_index_og,
                    lfp_data[i, :],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                resampled[i, :] = interp_func(new_time_index)
            
            resampled = resampled.T  # Transpose for standard orientation
        
        print(f"Resampled probe")
        return resampled, new_time_index
    
    def find_peak_ripple_channel(self, lfp_ca1, ca1_chans, filter_ripple_band_func):
        """
        Finds the CA1 channel with the highest ripple power.
        
        Parameters
        ----------
        lfp_ca1 : numpy.ndarray
            LFP data from CA1 channels
        ca1_chans : numpy.ndarray
            Indices of CA1 channels
        filter_ripple_band_func : function
            Function to filter signal to ripple band
            
        Returns
        -------
        tuple
            (peak_channel_index, peak_channel_id, peak_channel_raw_lfp)
        """
        
        lfp_ca1_rippleband = filter_ripple_band_func(lfp_ca1)
        highest_rip_power = np.abs(signal.hilbert(lfp_ca1_rippleband)) ** 2
        highest_rip_power = highest_rip_power.max(axis=0)
        
        peak_channel_idx = highest_rip_power.argmax()
        peak_channel_id = ca1_chans[peak_channel_idx]
        peak_channel_raw_lfp = lfp_ca1[:, peak_channel_idx]
        print(f"Finding channel with highest ripple power...")
        return peak_channel_idx, peak_channel_id, peak_channel_raw_lfp
    
    def process_probe(self, probe_idx, filter_ripple_band_func=None):
        """
        Processes a single probe completely.
        
        Parameters
        ----------
        probe_idx : int
            Index of the probe in the probelist
        filter_ripple_band_func : function, optional
            Function to filter for ripple band
            
        Returns
        -------
        dict
            Dictionary with processing results
        """
        if self.probelist is None:
            self.get_probe_ids_and_names()
            
        # Step 1: Load channels
        channels, probe_name, probe_id = self.load_channels(probe_idx)
        
        # Step 2: Check for CA1 channels
        if self.br is not None and not self.has_ca1_channels(channels):
            return None
            
        # Step 3: Load bin file
        bin_file = self.load_bin_file(probe_name)
        if bin_file is None:
            return None
            
        # Step 4: Read the data
        print(f"Reading LFP data for probe {probe_id}...")
        sr = spikeglx.Reader(bin_file)
        
        # Step 5: Create time index
        lfp_time_index_og = self.create_time_index(sr, probe_id)
        
        # Step 6: Extract raw data
        raw, fs_from_sr = self.extract_raw_data(sr)
        del sr  # Free memory
        
        # Step 7: Destripe data
        print(f"Destripping LFP data for probe {probe_id}...")
        destriped = self.destripe_data(raw, fs_from_sr)
        del raw  # Free memory
        
        # Step 8: Get CA1 channels
        lfp_ca1, ca1_chans = self.get_ca1_channels(channels, destriped)
        
        # Step 9: Get non-hippocampal control channels for artifact detection
        print(f"Getting control channels for artifact detection probe {probe_id}...")
        control_data, control_channels = self.get_non_hippocampal_channels(channels, destriped)
        
        # Step 10: Resample CA1 channels to 1500 Hz
        print(f"Resampling CA1 channels to 1.5kHz probe {probe_id}...")
        lfp_ca1, lfp_time_index = self.resample_signal(lfp_ca1, lfp_time_index_og, 1500.0)
        
        # Step 11: Resample control channels
        outof_hp_chans_lfp = []
        for channel_data in control_data:
            # Reshape to 2D array with shape (1, n_samples)
            channel_data = channel_data.reshape(1, -1)
            
            # Resample
            lfp_control, _ = self.resample_signal(channel_data, lfp_time_index_og, 1500.0)
            
            # Append to list, ensuring correct shape
            outof_hp_chans_lfp.append(lfp_control[:, None])
            del lfp_control  # Free memory
        
        del destriped  # Free memory for large array
        del control_data  # Free memory
        
        # Step 12: Find channel with highest ripple power if function provided
        if filter_ripple_band_func is not None:
            peak_idx, peak_id, peak_lfp = self.find_peak_ripple_channel(
                lfp_ca1, ca1_chans, filter_ripple_band_func
            )
            
            # Create a channel ID string for naming files
            this_chan_id = f"channelsrawInd_{peak_id}"
        else:
            peak_idx = None
            peak_id = None
            peak_lfp = None
            this_chan_id = None
            
        # Collect results
        results = {
            'probe_id': probe_id,
            'probe_name': probe_name,
            'lfp_ca1': lfp_ca1,
            'lfp_time_index': lfp_time_index,
            'channels': channels,
            'ca1_chans': ca1_chans,
            'control_lfps': outof_hp_chans_lfp,
            'control_channels': control_channels,
            'peak_ripple_chan_idx': peak_idx,
            'peak_ripple_chan_id': peak_id,
            'peak_ripple_chan_raw_lfp': peak_lfp,
            'chan_id_string': this_chan_id
        }
        
        return results
    
    def cleanup(self):
        """
        Cleans up resources to free memory.
        """
        self.data_files = None